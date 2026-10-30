import fs from "node:fs";
import path from "node:path";

const packFloat32 = (values: Float32Array): Buffer => {
  const buffer = Buffer.alloc(values.length * 4);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setFloat32(valueIndex * 4, values[valueIndex] as number, true);
  }

  return buffer;
};

const packFloat64 = (values: number[]): Buffer => {
  const buffer = Buffer.alloc(values.length * 8);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setFloat64(valueIndex * 8, values[valueIndex] as number, true);
  }

  return buffer;
};

const packUint16 = (values: number[]): Buffer => {
  const buffer = Buffer.alloc(values.length * 2);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setUint16(valueIndex * 2, values[valueIndex] as number, true);
  }

  return buffer;
};

const compareFiles = (
  expectedDirectory: string,
  actualDirectory: string,
  names: string[],
): string | null => {
  for (const fileName of names) {
    const expectedContent = fs.readFileSync(path.join(expectedDirectory, fileName));
    const actualContent = fs.readFileSync(path.join(actualDirectory, fileName));

    if (!expectedContent.equals(actualContent)) {
      return `${fileName} differs`;
    }
  }

  return null;
};

const compareDirectories = (reference: string, candidate: string): string | null => {
  const referenceFiles = fs.readdirSync(reference).sort();
  const candidateFiles = fs.readdirSync(candidate).sort();

  if (referenceFiles.join(",") !== candidateFiles.join(",")) {
    return `file lists differ: ${referenceFiles.join(", ")} against ${candidateFiles.join(", ")}`;
  }

  return compareFiles(reference, candidate, referenceFiles);
};

export { compareDirectories, compareFiles, packFloat32, packFloat64, packUint16 };
