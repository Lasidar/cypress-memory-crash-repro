Feature: Verify news items on Yahoo are visible

  Scenario Outline: Verify that items under category <category> are visible
    When I navigate to yahoo
    And I open the "<category>" section
    Then The <number_of_items> items on the page exist and can be navigated to

    Examples:
        | category | number_of_items |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |
        | Entertainment | 25 |
        | News | 25 |