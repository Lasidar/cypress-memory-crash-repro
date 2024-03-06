import { When, Then } from "@badeball/cypress-cucumber-preprocessor";

When("I navigate to yahoo", () => {
    cy.visit("https://ca.yahoo.com/");
});

When("I open the {string} section", (category) => {
    cy.findByRole("link", {name: category}).should("be.visible").click({force: true});
});

Then("The {int} items on the page exist and can be navigated to", (numberOfItems) => {
    /* Infinite scroll, so load some additional articles */
    cy.scrollTo("bottom");
    cy.scrollTo("bottom");
    cy.scrollTo("bottom");

    /* Check the first numberOfItems items  */
    for (let i = 0; i < numberOfItems; i++) {
        cy.get("[data-test-locator='stream-items']").children().eq(i).scrollIntoView().should("exist").click({force: true});
        cy.get("body").type("{esc}"); // Exit the article
        for (let j = 0; j < 50; j++) {
            cy.log("Random additional data injected into the runner log");
        }
    }
});