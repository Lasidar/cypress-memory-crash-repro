import { defineConfig } from "cypress";

export default defineConfig({
  redirectionLimit: 1000,
  env: {
    failSilently: false,
  },
  numTestsKeptInMemory: 0,
  retries: {
    runMode: 2,
    openMode: 0,
  },
  viewportWidth: 1024,
  viewportHeight: 768,
  video: true,
  videoCompression: 20,
  trashAssetsBeforeRuns: true,
  e2e: {
    setupNodeEvents(on, config) {
      return require("./cypress/plugins/index")(on, config);
    },
    specPattern: "cypress/tests/*.feature",
    experimentalMemoryManagement: true,
    experimentalModifyObstructiveThirdPartyCode: true,
  },
});
