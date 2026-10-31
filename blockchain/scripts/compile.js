const fs = require("fs");
const path = require("path");
const solc = require("solc");

const contractPath = path.join(__dirname, "..", "contracts", "SkillVerification.sol");
const source = fs.readFileSync(contractPath, "utf8");

const input = {
  language: "Solidity",
  sources: {
    "SkillVerification.sol": { content: source },
  },
  settings: {
    optimizer: { enabled: true, runs: 200 },
    outputSelection: {
      "*": {
        "*": ["abi", "evm.bytecode.object", "evm.gasEstimates"],
      },
    },
  },
};

const output = JSON.parse(solc.compile(JSON.stringify(input)));

if (output.errors) {
  const fatal = output.errors.filter((e) => e.severity === "error");
  output.errors.forEach((e) => console.error(e.formattedMessage));
  if (fatal.length) process.exit(1);
}

const contract = output.contracts["SkillVerification.sol"]["SkillVerification"];
const artifact = {
  abi: contract.abi,
  bytecode: "0x" + contract.evm.bytecode.object,
  gasEstimates: contract.evm.gasEstimates,
};

const outDir = path.join(__dirname, "..", "build");
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "SkillVerification.json"), JSON.stringify(artifact, null, 2));

console.log("Compiled OK. Bytecode length:", contract.evm.bytecode.object.length / 2, "bytes");
console.log("Deployment gas estimate:", contract.evm.gasEstimates.creation);
