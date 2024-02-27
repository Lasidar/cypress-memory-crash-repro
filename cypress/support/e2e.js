// ***********************************************************
// This example support/index.js is processed and
// loaded automatically before your test files.
//
// This is a great place to put global configuration and
// behavior that modifies Cypress.
//
// You can change the location of this file or turn off
// automatically serving support files with the
// 'supportFile' configuration option.
//
// You can read more here:
// https://on.cypress.io/configuration
// ***********************************************************

import "@testing-library/cypress/add-commands";

Cypress.config("defaultCommandTimeout", 20000);
Cypress.config("requestTimeout", 10000);

Cypress.on("uncaught:exception", () => {
  return false;
});

const origLog = Cypress.log;
Cypress.log = function (opts, ...other) {
  if ((opts.displayName === "fetch" || opts.displayName === "xhr") && routes.some((route) => opts?.url.search(route) != -1)) {
    return;
  }
  return origLog(opts, ...other);
};

// This block allows to hide XHR and Fetch console logs which have no sense for testing and only pollute the logs
Cypress.on("log:added", (ev) => {
  if (
    (ev?.displayName === "xhr" || ev?.displayName === "fetch") &&
    routes.some((route) => ev?.consoleProps?.props?.URL && ev.consoleProps.props.URL.search(route) != -1)
  ) {
    const el = Array.from(window.top.document.querySelectorAll(".command-wrapper")).slice(-1)[0];
    if (el) {
      el.parentElement.style.display = "none";
    }
  }
});