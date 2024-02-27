/// <reference types="cypress" />
const cucumber = require("cypress-cucumber-preprocessor").default;
const browserify = require("@cypress/browserify-preprocessor");

module.exports = (on, config) => {
  const options = {
    ...browserify.defaultOptions,
    typescript: require.resolve("typescript"),
  };

  on("file:preprocessor", cucumber(options));

  // eslint-disable-next-line @typescript-eslint/default-param-last
  on("before:browser:launch", (browser = {}, launchOptions) => {
    if (browser.family === "chromium") {
      launchOptions.args.push("--disable-dev-shm-usage");
      return launchOptions;
    } else if (browser.family === "firefox") {
      launchOptions.preferences["pdfjs.enabledCache.state"] = false;
      launchOptions.preferences["pdfjs.migrationVersion"] = 2;
      return launchOptions;
    }
  });

  return config;
};