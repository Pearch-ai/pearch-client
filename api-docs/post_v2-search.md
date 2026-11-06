## Search by coordinates and radius

In Fast search, you can specify a center and radius directly in `search_requirements`. Include the latitude, longitude, and distance unit in the requirement text, and set `must_have` to `true` when the location is mandatory.

For example, find software engineers within 10 miles of the specified coordinates in San Francisco:

```json
{
  "type": "fast",
  "limit": 5,
  "search_requirements": [
    {
      "search_requirement": "Currently works as a software engineer",
      "must_have": true
    },
    {
      "search_requirement": "Located within 10 miles of latitude 37.7749, longitude -122.4194",
      "must_have": true
    }
  ]
}
```

For a structured geographic filter, pass the same center and radius in `custom_filters.location_coordinates`:

```json
{
  "type": "fast",
  "limit": 5,
  "search_requirements": [
    {
      "search_requirement": "Currently works as a software engineer",
      "must_have": true
    }
  ],
  "custom_filters": {
    "location_coordinates": [
      {"lat": 37.7749, "lng": -122.4194, "radius_miles": 10}
    ]
  }
}
```

`radius_miles` is measured in miles and must be greater than 0 and at most 500. Multiple areas in `location_coordinates` are combined with OR. Different custom-filter fields are combined with AND: adding `locations: ["San Francisco"]` restricts results to the intersection of that location and the radius area. Leave out the city filter when you want the full radius, including nearby places outside the city.

Geographic filtering uses the location recorded for each profile, which may represent a city or region rather than a precise address.

<Cards columns={2}>
  <Card title="PRICING" href="https://pearch.ai/pricing" icon="fa-money-bill" target="_blank">
    CHECK HOW OUR CREDIT SYSTEM WORKS
  </Card>

  <Card title="KNOWLEDGE BASE" href="https://pearch.ai/articles" icon="fa-book" target="_blank">
    Insights and Research from PEARCH AI team
  </Card>
</Cards>
<BookDemoButton href="https://calendly.com/all_pearch/15-min-intro" label="Book a demo" sublabel="15 minutes • Calendly" size="lg" fullWidth={false} align="center" />
