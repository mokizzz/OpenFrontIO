
import { AStar, PathFindResultType } from "./AStar";
import { GameMap, TileRef } from "../game/GameMap";
import { GameMapAdapter } from "./MiniAStar";
import { SerialAStar } from "./SerialAStar";

export type Cluster = {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  entrances: TileRef[];
};

export type AbstractNode = {
  tile: TileRef;
  clusterId: number;
  isEntrance: boolean;
};

export type AbstractEdge = {
  from: AbstractNode;
  to: AbstractNode;
  cost: number;
  level: number;
};

class BoundedGameMapAdapter extends GameMapAdapter {
  constructor(
    private readonly baseGameMap: GameMap,
    waterPath: boolean,
    private readonly bounds: { x: number; y: number; width: number; height: number },
  ) {
    super(baseGameMap, waterPath);
  }

  neighbors(node: TileRef): TileRef[] {
    const originalNeighbors = super.neighbors(node);
    return originalNeighbors.filter((neighbor) => {
      const nx = this.baseGameMap.x(neighbor);
      const ny = this.baseGameMap.y(neighbor);
      return (
        nx >= this.bounds.x &&
        nx < this.bounds.x + this.bounds.width &&
        ny >= this.bounds.y &&
        ny < this.bounds.y + this.bounds.height
      );
    });
  }
}

export class HPAStar implements AStar<TileRef> {
  private clusters: Map<number, Cluster> = new Map();
  private abstractGraph: Map<string, AbstractEdge[]> = new Map();
  private readonly abstractNodes: Map<string, AbstractNode> = new Map();
  private readonly clusterSize: number;
  private readonly clustersX: number;

  private isInitialized = false;
  private currentPath: TileRef[] | null = null;
  private isPreprocessed = false;
  private initializationGenerator: Generator<void, void, void> | null = null;

  constructor(
    private readonly gameMap: GameMap,
    private readonly src: TileRef,
    private readonly dst: TileRef,
    clusterSize = 10,
    private readonly adapter: GameMapAdapter,
  ) {
    this.clusterSize = clusterSize;
    this.clustersX = Math.ceil(this.gameMap.width() / this.clusterSize);
  }

  compute(): PathFindResultType {
    if (!this.isInitialized) {
      this.initializationGenerator ??= this.initialize();

      const result = this.initializationGenerator.next();
      if (!result.done) {
        return PathFindResultType.Pending;
      }

      this.isInitialized = true;
      this.isPreprocessed = true;
      this.initializationGenerator = null;
    }

    return this.computeQuery();
  }

  computeQuery(): PathFindResultType {
    if (!this.isPreprocessed) {
      throw new Error("HPA* must be preprocessed before querying");
    }

    if (this.currentPath) {
      return PathFindResultType.Completed;
    }

    try {
      const { startKey, goalKey } = this.insertStartAndGoal();
      const abstractPathKeys = this.searchAbstractGraph(startKey, goalKey);

      if (abstractPathKeys.length === 0) {
        return PathFindResultType.PathNotFound;
      }

      const abstractPath = abstractPathKeys
        .map((key) => this.abstractNodes.get(key))
        .filter((node): node is AbstractNode => node !== undefined);

      this.currentPath = this.refinePath(abstractPath);
      return PathFindResultType.Completed;
    } catch (error) {
      return PathFindResultType.PathNotFound;
    }
  }

  // Methods for sharing preprocessed data
  getAbstractGraph(): Map<string, AbstractEdge[]> {
    return this.abstractGraph;
  }

  getClusters(): Map<number, Cluster> {
    return this.clusters;
  }

  getClusterCount(): number {
    return this.clusters.size;
  }

  setPreprocessedData(abstractGraph: Map<string, AbstractEdge[]>, clusters: Map<number, Cluster>): void {
    this.abstractGraph = new Map(abstractGraph);
    this.clusters = new Map(clusters);
    this.isPreprocessed = true;
    this.isInitialized = true;

    this.abstractNodes.clear();
    for (const edges of this.abstractGraph.values()) {
      for (const edge of edges) {
        this.abstractNodes.set(this.getNodeKeyFromNode(edge.from), edge.from);
        this.abstractNodes.set(this.getNodeKeyFromNode(edge.to), edge.to);
      }
    }
  }

  reconstructPath(): TileRef[] {
    return this.currentPath ?? [];
  }

  private *initialize(): Generator<void, void, void> {
    // buildClusters
    const mapWidth = this.gameMap.width();
    const mapHeight = this.gameMap.height();

    let clusterId = 0;
    for (let y = 0; y < mapHeight; y += this.clusterSize) {
      for (let x = 0; x < mapWidth; x += this.clusterSize) {
        const cluster: Cluster = {
          entrances: [],
          height: Math.min(this.clusterSize, mapHeight - y),
          id: clusterId++,
          width: Math.min(this.clusterSize, mapWidth - x),
          x,
          y,
        };
        this.clusters.set(cluster.id, cluster);
      }
      yield; // Yield after each row of clusters
    }

    // Build entrances and inter-edges simultaneously to ensure symmetry
    yield* this.buildEntrancesAndInterEdges();

    // Build intra-cluster edges (must be after entrance detection)
    const clustersForGraph = Array.from(this.clusters.values());
    for (const cluster of clustersForGraph) {
      this.buildIntraClusterEdges(cluster);
      yield; // Yield after each cluster
    }
  }

  private *buildEntrancesAndInterEdges(): Generator<void, void, void> {
    for (const cluster of this.clusters.values()) {
      // Check border to the RIGHT
      const rightClusterId = cluster.id + 1;
      // Ensure the right cluster is on the same row
      if ((cluster.id % this.clustersX) < (this.clustersX - 1) && this.clusters.has(rightClusterId)) {
        const rightCluster = this.clusters.get(rightClusterId);
        if (rightCluster) {
          this.processSharedBorder(cluster, rightCluster, "horizontal");
        }
      }

      // Check border BELOW
      const belowClusterId = cluster.id + this.clustersX;
      if (this.clusters.has(belowClusterId)) {
        const belowCluster = this.clusters.get(belowClusterId);
        if (belowCluster) {
          this.processSharedBorder(cluster, belowCluster, "vertical");
        }
      }
    }
    yield;
  }

  private processSharedBorder(
    c1: Cluster,
    c2: Cluster,
    orientation: "horizontal" | "vertical",
  ): void {
    let currentEntrance: { tile1: TileRef; tile2: TileRef }[] = [];

    const length = orientation === "horizontal"
      ? Math.min(c1.height, c2.height)
      : Math.min(c1.width, c2.width);

    for (let i = 0; i < length; i++) {
      let tile1: TileRef;
      let tile2: TileRef;

      if (orientation === "horizontal") {
        // Right border of c1, Left border of c2
        tile1 = this.gameMap.ref(c1.x + c1.width - 1, c1.y + i);
        tile2 = this.gameMap.ref(c2.x, c2.y + i);
      } else {
        // Bottom border of c1, Top border of c2
        tile1 = this.gameMap.ref(c1.x + i, c1.y + c1.height - 1);
        tile2 = this.gameMap.ref(c2.x + i, c2.y);
      }

      const isPassable = this.adapter.isTraversable(tile1, tile2);

      if (isPassable) {
        currentEntrance.push({ tile1, tile2 });
      }

      if (!isPassable || i === length - 1) {
        if (currentEntrance.length > 0) {
          const entranceSegment1 = currentEntrance.map((e) => e.tile1);
          const entranceSegment2 = currentEntrance.map((e) => e.tile2);

          const transitionPoints1 = this.createTransitionPoints(entranceSegment1);
          const transitionPoints2 = this.createTransitionPoints(entranceSegment2);

          // Add transition points to BOTH clusters' entrance lists
          c1.entrances.push(...transitionPoints1);
          c2.entrances.push(...transitionPoints2);

          // Create inter-edges for each pair of transition points
          for (let k = 0; k < transitionPoints1.length; k++) {
            const t1 = transitionPoints1[k];
            const t2 = transitionPoints2[k];
            this.createInterClusterEdge(c1, t1, c2, t2);
          }
          currentEntrance = [];
        }
      }
    }
  }

  private createTransitionPoints(entranceSegment: TileRef[]): TileRef[] {
    const ENTRANCE_WIDTH_THRESHOLD = 6; // the default value in the paper

    if (entranceSegment.length === 0) {
      return [];
    }

    if (entranceSegment.length < ENTRANCE_WIDTH_THRESHOLD) {
      // Small entrance: create one transition point in the middle
      const middleIndex = Math.floor(entranceSegment.length / 2);
      return [entranceSegment[middleIndex]];
    } else {
      // Large entrance: create transition points at both ends
      return [
        entranceSegment[0],
        entranceSegment[entranceSegment.length - 1],
      ];
    }
  }

  private buildIntraClusterEdges(cluster: Cluster): void {
    const { entrances } = cluster;

    for (let i = 0; i < entrances.length; i++) {
      const from = entrances[i];
      const fromKey = this.getNodeKey(from, cluster.id, true);
      const fromNode: AbstractNode = {
        clusterId: cluster.id,
        isEntrance: true,
        tile: from,
      };

      this.abstractNodes.set(fromKey, fromNode);
      if (!this.abstractGraph.has(fromKey)) {
        this.abstractGraph.set(fromKey, []);
      }

      for (let j = i + 1; j < entrances.length; j++) {
        const to = entrances[j];
        const distance = this.calculateIntraClusterDistance(from, to, cluster);

        if (distance < Infinity) {
          const toKey = this.getNodeKey(to, cluster.id, true);
          const toNode: AbstractNode = {
            clusterId: cluster.id,
            isEntrance: true,
            tile: to,
          };

          this.abstractNodes.set(toKey, toNode);
          if (!this.abstractGraph.has(toKey)) {
            this.abstractGraph.set(toKey, []);
          }

          // Add bidirectional edges
          this.addBidirectionalEdge(fromKey, toKey, fromNode, toNode, distance);
        }
      }
    }
  }

  private createInterClusterEdge(
    cluster: Cluster,
    entrance: TileRef,
    neighborCluster: Cluster,
    neighbor: TileRef,
  ): void {
    const fromKey = this.getNodeKey(entrance, cluster.id, true);
    const toKey = this.getNodeKey(neighbor, neighborCluster.id, true);

    const fromNode: AbstractNode = {
      clusterId: cluster.id,
      isEntrance: true,
      tile: entrance,
    };

    const toNode: AbstractNode = {
      clusterId: neighborCluster.id,
      isEntrance: true,
      tile: neighbor,
    };

    this.abstractNodes.set(fromKey, fromNode);
    this.abstractNodes.set(toKey, toNode);

    const cost = this.adapter.cost(neighbor);

    if (!this.abstractGraph.has(fromKey)) {
      this.abstractGraph.set(fromKey, []);
    }
    if (!this.abstractGraph.has(toKey)) {
      this.abstractGraph.set(toKey, []);
    }

    // Create bidirectional edges
    this.addBidirectionalEdge(fromKey, toKey, fromNode, toNode, cost, this.adapter.cost(entrance));
  }

  private insertStartAndGoal(): { startKey: string; goalKey: string } {
    const srcCluster = this.getClusterContaining(this.src);
    const dstCluster = this.getClusterContaining(this.dst);

    if (!srcCluster || !dstCluster) {
      throw new Error("Source or destination not found in any cluster");
    }

    const startKey = this.getNodeKey(this.src, srcCluster.id, false);
    const goalKey = this.getNodeKey(this.dst, dstCluster.id, false);

    const startNode: AbstractNode = {
      clusterId: srcCluster.id,
      isEntrance: false,
      tile: this.src,
    };

    const goalNode: AbstractNode = {
      clusterId: dstCluster.id,
      isEntrance: false,
      tile: this.dst,
    };

    this.abstractNodes.set(startKey, startNode);
    this.abstractNodes.set(goalKey, goalNode);

    // Connect start to entrances of its cluster
    this.abstractGraph.set(startKey, []);
    for (const entrance of srcCluster.entrances) {
      const distance = this.calculateIntraClusterDistance(this.src, entrance, srcCluster);
      if (distance < Infinity) {
        const entranceKey = this.getNodeKey(entrance, srcCluster.id, true);
        const entranceNode: AbstractNode = {
          clusterId: srcCluster.id,
          isEntrance: true,
          tile: entrance,
        };

        this.abstractNodes.set(entranceKey, entranceNode);

        const edge: AbstractEdge = {
          cost: distance,
          from: startNode,
          level: 0,
          to: entranceNode,
        };

        const startEdges = this.abstractGraph.get(startKey);
        if (startEdges) startEdges.push(edge);
      }
    }

    // Connect entrances of goal cluster to goal
    this.abstractGraph.set(goalKey, []);
    for (const entrance of dstCluster.entrances) {
      const distance = this.calculateIntraClusterDistance(entrance, this.dst, dstCluster);
      if (distance < Infinity) {
        const entranceKey = this.getNodeKey(entrance, dstCluster.id, true);
        const entranceNode: AbstractNode = {
          clusterId: dstCluster.id,
          isEntrance: true,
          tile: entrance,
        };

        this.abstractNodes.set(entranceKey, entranceNode);

        if (!this.abstractGraph.has(entranceKey)) {
          this.abstractGraph.set(entranceKey, []);
        }

        const edge: AbstractEdge = {
          cost: distance,
          from: entranceNode,
          level: 0,
          to: goalNode,
        };

        const entranceEdges = this.abstractGraph.get(entranceKey);
        if (entranceEdges) entranceEdges.push(edge);
      }
    }

    return { goalKey, startKey };
  }

  private searchAbstractGraph(startKey: string, goalKey: string): string[] {
    const openSet = new Map<string, number>(); // nodeKey -> fScore
    const closedSet = new Set<string>();
    const cameFrom = new Map<string, string>();
    const gScore = new Map<string, number>();

    const startNode = this.abstractNodes.get(startKey);
    const goalNode = this.abstractNodes.get(goalKey);

    if (!startNode || !goalNode) {
      return [];
    }

    gScore.set(startKey, 0);
    openSet.set(startKey, this.heuristicAbstract(startNode, goalNode));

    while (openSet.size > 0) {
      // Find node with lowest fScore
      let currentKey: string | undefined;
      let lowestFScore = Infinity;

      for (const [nodeKey, fScore] of openSet) {
        if (fScore < lowestFScore) {
          currentKey = nodeKey;
          lowestFScore = fScore;
        }
      }

      if (!currentKey) break;

      // Check if we reached the goal
      if (currentKey === goalKey) {
        return this.reconstructAbstractPath(cameFrom, currentKey);
      }

      openSet.delete(currentKey);
      closedSet.add(currentKey);

      // Explore neighbors
      const edges = this.abstractGraph.get(currentKey) ?? [];

      for (const edge of edges) {
        const neighborKey = this.getNodeKeyFromNode(edge.to);

        if (closedSet.has(neighborKey)) continue;

        const tentativeGScore = (gScore.get(currentKey) ?? Infinity) + edge.cost;

        const currentNeighborScore = gScore.get(neighborKey);
        if (currentNeighborScore === undefined || tentativeGScore < currentNeighborScore) {
          cameFrom.set(neighborKey, currentKey);
          gScore.set(neighborKey, tentativeGScore);
          const fScore = tentativeGScore + this.heuristicAbstract(edge.to, goalNode);
          openSet.set(neighborKey, fScore);
        }
      }

    }

    return []; // No path found
  }

  private refinePath(abstractPath: AbstractNode[]): TileRef[] {
    if (abstractPath.length <= 1) {
      return abstractPath.map((node) => node.tile);
    }

    const detailedPath: TileRef[] = [];

    for (let i = 0; i < abstractPath.length - 1; i++) {
      const from = abstractPath[i];
      const to = abstractPath[i + 1];

      // If nodes are in the same cluster, use intra-cluster pathfinding
      if (from.clusterId === to.clusterId) {
        const cluster = this.clusters.get(from.clusterId);
        if (cluster) {
          const segmentPath = this.findDetailedPath(from.tile, to.tile, cluster);
          // Avoid duplicating the start node (except for the first segment)
          if (i > 0 && segmentPath.length > 0) {
            segmentPath.shift();
          }
          detailedPath.push(...segmentPath);
        }
      } else {
        // Inter-cluster connection (should be adjacent)
        if (i > 0) {
          detailedPath.push(to.tile);
        } else {
          detailedPath.push(from.tile, to.tile);
        }
      }
    }

    return detailedPath;
  }

  private getClusterContaining(tile: TileRef): Cluster | null {
    const x = this.gameMap.x(tile);
    const y = this.gameMap.y(tile);

    const clusterCol = Math.floor(x / this.clusterSize);
    const clusterRow = Math.floor(y / this.clusterSize);
    const clusterId = clusterRow * this.clustersX + clusterCol;

    return this.clusters.get(clusterId) ?? null;
  }

  private manhattan(a: TileRef, b: TileRef): number {
    const posA = this.adapter.position(a);
    const posB = this.adapter.position(b);
    return Math.abs(posA.x - posB.x) + Math.abs(posA.y - posB.y);
  }

  private calculateIntraClusterDistance(from: TileRef, to: TileRef, cluster: Cluster): number {
    try {
      const boundedAdapter = new BoundedGameMapAdapter(this.gameMap, this.adapter.isWaterPath, cluster);
      const serialAStar = new SerialAStar(from, to, 5000, 1, boundedAdapter);
      const result = serialAStar.compute();

      if (result === PathFindResultType.Completed) {
        return serialAStar.reconstructPath().length - 1;
      }
    } catch {
      // Fallback to Infinity if bounded search fails
    }
    return Infinity;
  }

  private findDetailedPath(from: TileRef, to: TileRef, cluster: Cluster): TileRef[] {
    try {
      const boundedAdapter = new BoundedGameMapAdapter(this.gameMap, this.adapter.isWaterPath, cluster);
      const serialAStar = new SerialAStar(from, to, 5000, 1, boundedAdapter);
      if (serialAStar.compute() === PathFindResultType.Completed) {
        return serialAStar.reconstructPath();
      }
    } catch (error) {
      console.error(`[HPA*] findDetailedPath failed between nodes. Falling back to a straight line. Error: ${error}`);
    }
    // Fallback to direct path if bounded search fails
    return [from, to];
  }

  private heuristicAbstract(a: AbstractNode, b: AbstractNode): number {
    return this.manhattan(a.tile, b.tile);
  }

  private reconstructAbstractPath(cameFrom: Map<string, string>, currentKey: string): string[] {
    const path: string[] = [currentKey];

    while (cameFrom.has(currentKey)) {
      const nextKey = cameFrom.get(currentKey);
      if (!nextKey) break;
      currentKey = nextKey;
      path.unshift(currentKey);
    }

    return path;
  }

  private getNodeKey(tile: TileRef, clusterId: number, isEntrance: boolean): string {
    const x = this.gameMap.x(tile);
    const y = this.gameMap.y(tile);
    return `${x},${y},${clusterId},${isEntrance}`;
  }

  private getNodeKeyFromNode(node: AbstractNode): string {
    return this.getNodeKey(node.tile, node.clusterId, node.isEntrance);
  }

  private addBidirectionalEdge(
    fromKey: string,
    toKey: string,
    fromNode: AbstractNode,
    toNode: AbstractNode,
    forwardCost: number,
    backwardCost?: number,
  ): void {
    const edge1: AbstractEdge = {
      cost: forwardCost,
      from: fromNode,
      level: 0,
      to: toNode,
    };

    const edge2: AbstractEdge = {
      cost: backwardCost ?? forwardCost,
      from: toNode,
      level: 0,
      to: fromNode,
    };

    const fromEdges = this.abstractGraph.get(fromKey);
    const toEdges = this.abstractGraph.get(toKey);
    if (fromEdges) fromEdges.push(edge1);
    if (toEdges) toEdges.push(edge2);
  }
}

type PathData = {
  abstractGraph: Map<string, AbstractEdge[]>;
  clusters: Map<number, Cluster>;
  clusterSize: number;
  ready: boolean;
};

type GameData = {
  land: PathData | null;
  water: PathData | null;
};

export class HPACache {
  private static readonly gameData = new WeakMap<GameMap, GameData>();

  public static initForGame(gameMap: GameMap, waterPath: boolean): Promise<void> {
    if (!this.gameData.has(gameMap)) {
      this.gameData.set(gameMap, { land: null, water: null });
    }

    const gameData = this.gameData.get(gameMap);
    if (!gameData) {
      throw new Error("Game data not found");
    }
    const pathType: keyof GameData = waterPath ? "water" : "land";

    if (gameData[pathType]?.ready) {
      return Promise.resolve();
    }

    const clusterSize = this.calculateOptimalClusterSize(gameMap);

    const data: PathData = {
      abstractGraph: new Map<string, AbstractEdge[]>(),
      clusterSize,
      clusters: new Map<number, Cluster>(),
      ready: false,
    };
    gameData[pathType] = data;

    const tempHPAStar = new HPAStar(
      gameMap,
      gameMap.ref(0, 0),
      gameMap.ref(100, 100),
      clusterSize,
      new GameMapAdapter(gameMap, waterPath),
    );

    return new Promise((resolve) => {
      const computeStep = () => {
        const result = tempHPAStar.compute();
        if (result === PathFindResultType.Pending) {
          setTimeout(computeStep, 0);
        } else {
          data.abstractGraph = tempHPAStar.getAbstractGraph();
          data.clusters = tempHPAStar.getClusters();
          data.ready = true;
          resolve();
        }
      };
      computeStep();
    });
  }

  public static findPath(gameMap: GameMap, from: TileRef, to: TileRef, waterPath: boolean): TileRef[] | null {
    const gameData = this.gameData.get(gameMap);
    const pathType: keyof GameData = waterPath ? "water" : "land";
    const data = gameData?.[pathType];

    if (!data?.ready) {
      return null;
    }

    const hpaStar = new HPAStar(
      gameMap,
      from,
      to,
      data.clusterSize,
      new GameMapAdapter(gameMap, waterPath),
    );
    hpaStar.setPreprocessedData(data.abstractGraph, data.clusters);

    const result = hpaStar.computeQuery();
    if (result === PathFindResultType.Completed) {
      return hpaStar.reconstructPath();
    }
    return null;
  }

  public static isReady(gameMap: GameMap, waterPath: boolean): boolean {
    const gameData = this.gameData.get(gameMap);
    const pathType: keyof GameData = waterPath ? "water" : "land";
    return gameData?.[pathType]?.ready ?? false;
  }

  private static calculateOptimalClusterSize(gameMap: GameMap): number {
    const numClusters = 128;
    const recommandedSize = Math.round(Math.sqrt((gameMap.width() * gameMap.height()) / numClusters));
    return Math.max(32, Math.min(512, recommandedSize));
  }
}
