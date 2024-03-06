import { When, Then } from "@badeball/cypress-cucumber-preprocessor";

afterEach( () => {
    if (Cypress.browser.family === "chromium") {
        cy.window().then((window) => {
            let performanceInfo: any = window.performance;

            /* Compute the memory usage in MB */
            const usedMemory = Number(performanceInfo.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1);
            const absoluteMaxMemory = Number(performanceInfo.memory.jsHeapSizeLimit  / (1024 * 1024)).toFixed(1);
            const currentMaxMemory = Number(performanceInfo.memory.totalJSHeapSize  / (1024 * 1024)).toFixed(1);
            const percentUsedMax = Number(performanceInfo.memory.usedJSHeapSize * 100 / performanceInfo.memory.jsHeapSizeLimit).toFixed(1);
            const percentUsedActual = Number(performanceInfo.memory.usedJSHeapSize * 100 / performanceInfo.memory.totalJSHeapSize).toFixed(1);
            cy.task("log", `usedJSHeapSize: ${usedMemory}MB, ${percentUsedActual}% of allowed max ${currentMaxMemory}MB, ${percentUsedMax}% of allowed max ${absoluteMaxMemory}MB`);
        })
    }
})

Then("The tests intentionally fail", () => {
    expect(false).to.eq(true);
});