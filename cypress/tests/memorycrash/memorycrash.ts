import { When } from "cypress-cucumber-preprocessor/steps";

When("I navigate to yahoo", () => {
    cy.visit("https://ca.yahoo.com/?guccounter=1");
});

When("I repeatedly navigate to different pages", () => {
    for (let i = 0; i < 500; i++) {
        cy.findByRole("link", {name: "News"}).click();
        cy.get("[data-test-locator='stream-items']").each(($item) => {
            cy.wrap($item).should("be.visible");
        })

        cy.findByRole("link", {name: "Entertainment"}).click();
        cy.get("[data-test-locator='stream-items']").each(($item) => {
            cy.wrap($item).should("be.visible");
        })

        cy.findByRole("link", {name: "World"}).click();
        cy.get("[data-test-locator='stream-items']").each(($item) => {
            cy.wrap($item).should("be.visible");
        })
    }
});