/// <reference types="cypress" />
import { defineConfig } from "cypress";
import { addCucumberPreprocessorPlugin } from "@badeball/cypress-cucumber-preprocessor";
import { preprocessor } from "@badeball/cypress-cucumber-preprocessor/browserify";

async function setupNodeEvents(
  on: Cypress.PluginEvents,
  config: Cypress.PluginConfigOptions
): Promise<Cypress.PluginConfigOptions> {

  // This is required for the preprocessor to be able to generate JSON reports after each run, and more,
  await addCucumberPreprocessorPlugin(on, config);

  on(
    "file:preprocessor",
    preprocessor(config, {
      typescript: require.resolve("typescript"),
    })
  );

  // eslint-disable-next-line @typescript-eslint/default-param-last
  on("before:browser:launch", (browser, launchOptions) => {
    if (browser.family === "chromium") {
    //   launchOptions.args.push("--disable-dev-shm-usage");
      launchOptions.args.push('--enable-precise-memory-info')
      return launchOptions;
    }
  });

  // in plugins file
  on("task", {
    log(message) {
      console.log(`      INFO: ${message}`);
      return null;
    },
  });

  // Make sure to return the config object as it might have been modified by the plugin.
  return config;
}

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
    specPattern: "cypress/tests/*.{ts,feature}",
    experimentalMemoryManagement: true,
    experimentalModifyObstructiveThirdPartyCode: true,
    setupNodeEvents
  },
});
