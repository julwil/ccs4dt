# ccs4dt

Correlation Engine for Livealytics

---

## Installation

[Docker](https://docs.docker.com/get-docker/) and [Docker-compose](https://docs.docker.com/compose/install/) are
required.

1. Clone repository at `git@github.com:mtornow/ccs4dt.git`
2. Go into project root folder `cd ccs4dt/`
3. Run `docker-compose up`

---

## Getting Started

### REST API

API is exposed at [http://localhost:5000](http://localhost:5000) Find API
documentation [here](https://app.swaggerhub.com/apis-docs/julwil/ccs4dt/1.0.0).

---

### CoreDB

Stores configuration and metadata:

- Configuration for each location: How many sensors, where they are located, etc
- Metadata on Input/Output Batches (scheduled, processing, finished, failed)

---

### InfluxDB

Stores the actual time-series data provided by the InputBatch

InfluxDB UI is exposed at [http://localhost:8086](http://localhost:8086)

- username:  ccs4dt
- password: ccs4dt1234

---

## Tests

All tests are in the project's `/test` folder

```
ccs4dt
│   ...
└───tests
│   └───integration
│   └───unit
│   ...
```

prefix all tests with `test_` and place them in `/tests/integration` or `/tests/unit` folder.

Run all tests with `docker run api pytest` or `docker exec api pytest`

---

## Documentation
### REST API
OpenAPI v3 specification can be found [here](https://app.swaggerhub.com/apis-docs/julwil/ccs4dt/1.0.0)
### Backend
1. Run `docker exec api /bin/bash scripts/docs/generate.sh` to generate documentation
2. Open the generated HTML file at `docs/_build/html/index.html` in a browser
---