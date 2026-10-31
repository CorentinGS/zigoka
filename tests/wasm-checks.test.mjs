import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const ARTIFACTS = [
  { name: "infer.wasm", extraExports: [] },
  { name: "infer-safe.wasm", extraExports: [] },
  { name: "infer-profile.wasm", extraExports: ["forward_trunk"] },
  {
    name: "search.wasm",
    extraExports: [
      "moves_ptr",
      "position_ptr",
      "rules_area_score",
      "rules_dead_stones",
      "rules_encode",
      "rules_is_game_over",
      "rules_legal_moves",
      "rules_play",
      "search_best_move",
      "search_child_count",
      "search_child_move",
      "search_child_visits",
      "search_node_capacity",
      "search_nodes",
      "search_reused",
      "search_run",
      "search_start",
      "search_visits",
    ],
  },
];
const PUBLISHED_ARTIFACT = "infer.wasm";
const PUBLISHED_MODEL_ID = "moka-9";
const PUBLISHED_MEMORY_PAGES = 13;
const DIST_FILES = ["go-model.bin", "go-model.json", "infer.wasm", "worker.js"];
const GZIP_SIZE_CAP_BYTES = 32 * 1024;
const GZIP_LEVEL = 9;
const WASM_HEADER_BYTES = 8;
const MEMORY_SECTION_ID = 5;
const CODE_SECTION_ID = 10;
const LIMITS_HAS_MAXIMUM = 0x01;
const SIMD_PREFIX = 0xfd;
const MEMORY_GROW = [0x40, 0x00];
const MINIMUM_SIMD_PREFIX_COUNT = 128;

const EXPECTED_EXPORTS = [
  "__stack_pointer",
  "board_size",
  "expected_sha256_ptr",
  "feature_count",
  "infer",
  "input_ptr",
  "input_stride",
  "load_weights",
  "max_batch",
  "memory",
  "policy_moves",
  "policy_ptr",
  "value_ptr",
  "weights_capacity",
  "weights_ptr",
];

const createReader = (bytes, offset = 0) => {
  const reader = {
    offset,
    readByte: () => {
      reader.offset += 1;

      return bytes[reader.offset - 1];
    },
    readUnsignedLeb: () => {
      let result = 0;
      let shift = 0;
      let byte;

      do {
        byte = reader.readByte();
        result += (byte & 0x7f) * 2 ** shift;
        shift += 7;
      } while (byte & 0x80);

      return result;
    },
  };

  return reader;
};

const readSections = (bytes) => {
  const sections = new Map();
  const reader = createReader(bytes, WASM_HEADER_BYTES);

  while (reader.offset < bytes.length) {
    const id = reader.readByte();
    const size = reader.readUnsignedLeb();
    sections.set(id, bytes.subarray(reader.offset, reader.offset + size));
    reader.offset += size;
  }

  return sections;
};

const readMemoryLimits = (section) => {
  const reader = createReader(section);
  const count = reader.readUnsignedLeb();
  const flags = reader.readByte();
  const initial = reader.readUnsignedLeb();
  const maximum = flags & LIMITS_HAS_MAXIMUM ? reader.readUnsignedLeb() : null;

  return { count, initial, maximum };
};

const countSequence = (bytes, sequence) => {
  let count = 0;

  for (let index = 0; index + sequence.length <= bytes.length; index += 1) {
    if (sequence.every((byte, offset) => bytes[index + offset] === byte)) {
      count += 1;
    }
  }

  return count;
};

const modelIds = fs
  .readdirSync(path.join(rootDirectory, "zig-out", "models"))
  .filter((id) => fs.existsSync(path.join(rootDirectory, "zig-out", "models", id, "go-model.json")))
  .sort();

for (const id of modelIds) {
  for (const { name: artifact, extraExports } of ARTIFACTS) {
    const label = `${id}/${artifact}`;
    const bytes = fs.readFileSync(path.join(rootDirectory, "zig-out", id, artifact));
    const sections = readSections(bytes);

    test(`${label}: the code uses v128 instructions`, () => {
      const code = sections.get(CODE_SECTION_ID);

      assert.ok(
        countSequence(code, [SIMD_PREFIX]) >= MINIMUM_SIMD_PREFIX_COUNT,
        "too few 0xfd prefixes: simd128 is probably missing from the build",
      );
    });

    test(`${label}: the memory is declared with initial == maximum`, () => {
      const limits = readMemoryLimits(sections.get(MEMORY_SECTION_ID));

      assert.equal(limits.count, 1);
      assert.equal(limits.maximum, limits.initial);

      if (id === PUBLISHED_MODEL_ID && artifact.startsWith("infer") && artifact !== "infer-profile.wasm") {
        assert.equal(limits.initial, PUBLISHED_MEMORY_PAGES);
      }
    });

    test(`${label}: the code has no memory.grow (G5)`, () => {
      assert.equal(countSequence(sections.get(CODE_SECTION_ID), MEMORY_GROW), 0);
    });

    test(`${label}: the exports are exactly those of the ABI`, () => {
      const module = new WebAssembly.Module(bytes);
      const exports = WebAssembly.Module.exports(module)
        .map(({ name }) => name)
        .sort();

      assert.deepEqual(exports, [...EXPECTED_EXPORTS, ...extraExports].sort());
      assert.deepEqual(WebAssembly.Module.imports(module), []);
    });
  }
}

test(`${PUBLISHED_MODEL_ID}/${PUBLISHED_ARTIFACT}: the gzip size stays under the cap`, (t) => {
  const bytes = fs.readFileSync(
    path.join(rootDirectory, "zig-out", PUBLISHED_MODEL_ID, PUBLISHED_ARTIFACT),
  );
  const gzipBytes = zlib.gzipSync(bytes, { level: GZIP_LEVEL }).length;

  t.diagnostic(`${bytes.length} bytes, ${gzipBytes} after gzip (cap ${GZIP_SIZE_CAP_BYTES})`);
  assert.ok(gzipBytes <= GZIP_SIZE_CAP_BYTES, `${gzipBytes} bytes after gzip`);
});

test("dist/ holds exactly the published files", () => {
  assert.deepEqual(fs.readdirSync(path.join(rootDirectory, "dist")).sort(), DIST_FILES);
});
