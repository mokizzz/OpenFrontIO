import Benchmark from "benchmark";
import { GameMapAdapter } from "../../src/core/pathfinding/MiniAStar";
import { HPAStar } from "../../src/core/pathfinding/HPAStar";
import { PathFinder } from "../../src/core/pathfinding/PathFinding";
import { dirname } from "path";
import { fileURLToPath } from "url";
import { setup } from "../util/Setup";

const game = await setup(
  "giantworldmap",
  {},
  [],
  dirname(fileURLToPath(import.meta.url)),
);

const createAdapter = (waterPath: boolean) => new GameMapAdapter(game.map(), waterPath);

const clusterSizes = [64, 128, 256, 512];
const testPaths = [
  { name: "top-left-to-bottom-right", src: game.ref(0, 0), dst: game.ref(4077, 1929) },
  { name: "hawaii-to-svalbard", src: game.ref(186, 800), dst: game.ref(2205, 52) },
  { name: "nearby-tiles", src: game.ref(100, 100), dst: game.ref(200, 200) },
];

console.log(`Map: ${game.map().width()}x${game.map().height()}`);

const preprocessedInstances = new Map<number, HPAStar>();
const preprocessingTimes = new Map<number, number>();

for (const clusterSize of clusterSizes) {
  const startTime = Date.now();

  const hpaStar = new HPAStar(
    game.map(),
    game.ref(0, 0),
    game.ref(100, 100),
    clusterSize,
    createAdapter(true),
  );

  hpaStar.compute();
  const preprocessTime = Date.now() - startTime;

  preprocessedInstances.set(clusterSize, hpaStar);
  preprocessingTimes.set(clusterSize, preprocessTime);

  console.log(`Cluster ${clusterSize}: ${preprocessTime}ms`);
}

const suite = new Benchmark.Suite();
const results = new Map<string, { hz: number; rme: number }>();

// Add HPA* tests
for (const clusterSize of clusterSizes) {
  for (const path of testPaths) {
    const testName = `hpa-${clusterSize}-${path.name}`;

    suite.add(testName, () => {
      const hpaStar = new HPAStar(
        game.map(),
        path.src,
        path.dst,
        clusterSize,
        createAdapter(true),
      );

      const preprocessed = preprocessedInstances.get(clusterSize)!;
      hpaStar.setPreprocessedData(
        preprocessed.getAbstractGraph(),
        preprocessed.getClusters(),
      );

      hpaStar.computeQuery();
      hpaStar.reconstructPath();
    });
  }
}

// Add traditional A* tests for comparison
for (const path of testPaths) {
  const testName = `astar-${path.name}`;

  suite.add(testName, () => {
    PathFinder.Mini(game, 10000000, true, 1).nextTile(path.src, path.dst);
  });
}

suite
  .on("cycle", (event: any) => {
    const test = event.target;
    results.set(test.name, { hz: test.hz, rme: test.stats.rme });
    console.log(String(event.target));
  })
  .on("complete", function () {
    analyzeResults();
    generateRecommendations();
  })
  .run({ async: true });

function analyzeResults() {
  console.log("\nComparison:");

  for (const path of testPaths) {
    console.log(`\n${path.name}:`);

    const astarResult = results.get(`astar-${path.name}`);
    if (astarResult) {
      console.log(`  Traditional A*: ${astarResult.hz.toFixed(1)} ops/sec`);
    }

    let bestHPA = 0;
    let bestSize = 0;

    for (const clusterSize of clusterSizes) {
      const hpaResult = results.get(`hpa-${clusterSize}-${path.name}`);
      if (hpaResult && hpaResult.hz > bestHPA) {
        bestHPA = hpaResult.hz;
        bestSize = clusterSize;
      }

      if (hpaResult) {
        const speedup = astarResult ? (hpaResult.hz / astarResult.hz).toFixed(1) : "N/A";
        console.log(`  HPA* ${clusterSize}: ${hpaResult.hz.toFixed(1)} ops/sec (${speedup}x)`);
      }
    }

    if (bestSize > 0) {
      console.log(`  Best HPA*: cluster ${bestSize}`);
    }
  }
}

function generateRecommendations() {
  const mapSize = game.map().width() * game.map().height();
  const adaptiveSize = Math.max(32, Math.min(512, Math.round(Math.sqrt(mapSize / 128))));

  console.log(`\nAdaptive recommendation for ${(mapSize / 1000000).toFixed(1)}M tiles: cluster ${adaptiveSize}`);

  const shortResult = results.get("hpa-128-nearby-tiles");
  const astarShortResult = results.get("astar-nearby-tiles");

  if (shortResult && astarShortResult && shortResult.hz < astarShortResult.hz * 0.7) {
    console.log("Note: Use A* for distances <750, HPA* for ≥750");
  }
}
