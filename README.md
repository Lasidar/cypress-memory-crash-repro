To run these examples, change the desired browser with `--browser` and either enable or disable video using the `-c video=` parameter.

**Chrome**:

- **CrashingRun_WithoutVideo_Chrome.txt** - `npx cypress run --e2e --browser chrome -c video=false --spec cypress/tests/YahooItemsFail.feature` 
- **CrashingRun_WithVideo_Chrome.txt** - `npx cypress run --e2e --browser chrome -c video=true --spec cypress/tests/YahooItemsFail.feature` 

**Electron**:

- **CrashingRun_WithoutVideo_Electron.txt** - `npx cypress run --e2e --browser electron -c video=false --spec cypress/tests/YahooItemsFail.feature`
- **CrashingRun_WithVideo_Electron.txt** - `npx cypress run --e2e --browser electron -c video=true --spec cypress/tests/YahooItemsFail.feature`

**Firefox**:

- **NonCrashingRun_WithoutVideo_Firefox.txt** - `npx cypress run --e2e --browser firefox -c video=false --spec cypress/tests/YahooItemsFail.feature`
- *There is no run with video since video capture is not supported in Firefox*

Observations:
- There does not seem to be a significant difference in memory utilization between runs recording video and those not recording video. Ex: after the same test iteration in Chrome, the values of `usedJSHeapSize` with and without video were `781.8MB` and `778.4MB` respectively. 
- The memory does not seem to return to a baseline value after a test runs. The value of `usedJSHeapSize` steadily increases throughout the test run. This suggest a possible memory leak or issue with the memory being cleared between tests. Furthermore, `numTestsKeptInMemory` is set to `0`.
- The tests are not significant in size and would be typical for the amount data one of our longer running E2E tests might generate.
- experimentalMemoryManagement is enabled, and does not seem to have helped with the memory being cleared back to a baseline.

Notes: 
- Since the Cypress memory crash issue (https://github.com/cypress-io/cypress/issues/27415) appears to be anecdotally more prevalent when there are failing tests in the run, this example has been deliberately designed with failing tests.
- For tests in Chromimum browsers, an `afterEach` hook  prints the memory utilization information (usedJSHeapSize and totalJSHeapSize) after each test completes.
- Logs for each of these variations has already been provided under `/logs`. These examples were executed locally on a Macbook Pro M2 with 16GB of RAM and a fast SSD running Ventura 13.6.3.
- Although this configuration uses Cypress 13.0.0 which is representative of our current environment, the tests were also run against the latest version 13.6.6 and the same issues were observed.