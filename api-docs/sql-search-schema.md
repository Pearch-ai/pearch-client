This is the complete filter reference for [SQL search](./post_v2-search-sql). Put these predicates in `sql_filter`. The optional `query` parameter ranks the results; omit it for a filters-only search.

The SQL schema is mostly a subset of the returned profile schema. Only fields usable in SQL are listed here, including fields that support presence checks. Some filter-only helpers are also available. Other fields may appear in returned profiles without being available for filtering.

## Profile-shaped SQL schema

One indexed document represents one person. SQL exposes grouped experiences like the returned Pearch profile: `experiences[]` contains `company_info` and `company_roles[]`. Company attributes belong to the company group; titles, dates, durations, and role locations belong to `company_roles`.

```json
{
  "summary": "Backend engineer",
  "inferred_languages": [{"language": "English"}],
  "experiences": [{
    "company_info": {"domain": "stripe.com", "num_employees": 1000},
    "company_roles": [{"title": "Engineer", "start_date": "2022-01-01", "end_date": null}]
  }]
}
```

> 📘 Profile shape, indexed values
>
> Field names and nesting follow the response where possible. SQL still operates on indexed values, not the formatted response JSON. `summary` maps to indexed summary text and `inferred_languages` to indexed inferred languages. Dates use the documented SQL types. Contact values remain unavailable. A response-only field absent from these tables cannot be filtered.

Company metadata in the current index is copied onto role records. A company-and-role predicate tests the metadata copy attached to that matching role. When those copies disagree, the company metadata displayed in the response can come from a different role. This interface does not normalize conflicting company snapshots or guarantee equality with filtering the final formatted JSON.

## Supported SQL

Use a restricted BigQuery-style predicate, not an arbitrary query. A complete top-level `SELECT`, `ORDER BY`, joins to external tables, or SQL writes are not accepted. `SELECT` is permitted inside the supported `EXISTS` and aggregate forms shown here.

| Operation | Example | Meaning |
| --- | --- | --- |
| Comparisons | `total_experience_years >= 5` | At least five years of overall experience. |
| Alternatives | `location_country_code IN ('US', 'CA')` | Located in the United States or Canada. |
| Inclusive range | `total_experience_years BETWEEN 5 AND 10` | Five to ten years of overall experience, including both endpoints. |
| Presence | `first_name IS NOT NULL` | The person's profile includes a first name. |
| Boolean composition | `A AND (B OR C)`, `NOT (A)` | Require A and either B or C; or exclude matches for A. A, B, and C stand for conditions. |
| Named place | `LOCATION('Athens, Greece', 'GR')` | Located in Athens, Greece. |
| Distance radius | `LOCATION('New York, United States', 'US', 50, 'km')` | Located within 50 km of New York's resolved center. |
| Text or whole-value search | `SEARCH(title, 'backend AND engineer')` | The person's headline mentions both backend and engineer. |
| Repeated records | `EXISTS (SELECT 1 FROM UNNEST(experiences) AS e WHERE ...)` | The person has worked at a company matching the conditions replacing `...`. |
| Role records | `CROSS JOIN UNNEST(c.company_roles) AS r` after `UNNEST(experiences) AS c` | Read job titles, dates, and other role details at that company. |
| Nested repeated records | `CROSS JOIN UNNEST(c.company_info.funding_rounds) AS fr` within a company scope | Read that company's funding rounds. |
| Supported aggregate | `SUM(r.duration_years)` over a company source and its role source | Total years spent in matching roles. |
| Typed date/time | `r.start_date >= DATE '2022-01-01'` or `r.start_date >= TIMESTAMP '2022-01-01T00:00:00Z'` | A role started on or after January 1, 2022. |

Use single-quoted SQL strings, numeric literals for numbers, and `TRUE`/`FALSE` for booleans. Escape apostrophes by doubling them: `first_name = 'O''Brien'`. Use `IS NULL` and `IS NOT NULL` for missing values; `field = NULL` is not a presence test.

Provide one SQL condition in `sql_filter`. Combine conditions with `AND`, `OR`, and parentheses. Unsupported fields and constructs return HTTP 400.

### Names

Use `first_name`, `middle_name`, and `last_name` for name filters. Matching ignores case, trims and collapses whitespace, and removes Latin combining accents. For example, `Sánchez` matches `SANCHEZ`, and `Straße` matches `strasse`. Non-Latin scripts are preserved. Each condition matches the complete name component; a middle name can contain several words. These rules also apply to `IN` and `SEARCH`. Returned names retain their original spelling.

Find people named José Sánchez or José De Cózar using name filters without accents:

```sql
first_name = 'Jose'
AND last_name IN ('Sanchez', 'De Cozar')
```

Use `IS NULL` for a missing name; blank name filter values are rejected.

### Record presence

Top-level record arrays support `IS NOT NULL` to require at least one record with searchable data and `IS NULL` to require none.

Find people with at least one certification:

```sql
certifications IS NOT NULL
```

For certifications, a title or issuer is sufficient; a record containing only a certificate URL is not searchable. To constrain a specific certification, use `EXISTS` with conditions on its supported fields.

### SEARCH: text versus exact values

On fields marked **Field text SEARCH**, `SEARCH` matches terms, with stemming unless disabled in the table. Tokenized fields do not support exact `=` comparisons. On exact fields, including canonical string arrays, `SEARCH` matches whole indexed values. Canonical labels are not automatically resolved from arbitrary text.

Find backend engineers or developers with Python or Go skills and at least one skill listed on their profile:

```sql
SEARCH(title, 'backend AND (engineer OR developer)')
AND SEARCH(canonical_skills, 'Python OR Go')
AND skills IS NOT NULL
```

Use `AND`, `OR`, and parentheses inside a `SEARCH` string. To negate a search, write SQL `NOT SEARCH(...)`; do not put SQL conditions in the search string. Quoting a multiword term groups it, but tokenized search does not promise positional phrase matching. Use an exact canonical field when a whole-value match matters. Do not assume `LIKE`, wildcards, regular expressions, or arbitrary SQL functions are supported.

Some tokenized fields contribute to the document-wide `_text` index without having a field-specific text index. Those are marked **Document text only**. Field-specific `SEARCH` on `summary`, `skills`, an education's `major`, or a company's `technologies` is unsupported. Use `query`, a corresponding canonical field, or document-wide text search:

Find people whose profile text mentions both Python and distributed systems:

```sql
SEARCH(_text, 'Python AND distributed systems')
```

`_text` is a synthetic document-wide text field, not a returned profile property. A match can come from any contributing field, including another experience or education record. It does not establish that a term appears in a particular role, course, major, or company. Do not use it as a substitute for a same-record constraint. Presence tests on an indexed tokenized field are still supported.

### Keep related conditions in one scope

Find people currently employed at Stripe as engineers:

```sql
EXISTS (
  SELECT 1
  FROM UNNEST(experiences) AS c
  CROSS JOIN UNNEST(c.company_roles) AS r
  WHERE r.end_date IS NULL
    AND c.company_info.domain = 'stripe.com'
    AND SEARCH(r.title, 'engineer')
)
```

`r.end_date IS NULL` selects roles marked as current. Keeping the company and role conditions in the same `EXISTS` ensures that the engineering job is at Stripe. Separate `EXISTS` clauses could match an engineering job at another company and a different job at Stripe. Use the appropriate source alias for each field inside a subquery.

### Translation boundaries

Use at most one `company_roles` alias per company alias, in the same `SELECT` scope, through `CROSS JOIN UNNEST(c.company_roles) AS r`. A nested `EXISTS` or aggregate over an outer company's `company_roles`, multiple role aliases for that company, and offsets are rejected with HTTP 400. In particular, asking for two different roles at the same company is not supported. Splitting that requirement into independent `EXISTS` clauses would change its meaning.

Aggregates over company groups, such as `COUNT(*)` directly over `UNNEST(experiences)`, are rejected because the internal records count roles rather than distinct companies. Aggregate role records explicitly instead. Company-only positive and negative `EXISTS` filters are supported. `NOT EXISTS` over a company/role join excludes matching roles anywhere in the profile; it does not express a nested absence of roles within a selected company.

Flat role paths such as `e.title` after `UNNEST(experiences) AS e` are rejected. Use `r.title` after unnesting `e.company_roles`. Unsupported mappings are never approximated or dropped.

Exclude people who have ever worked at the company with domain `example.com`:

```sql
NOT EXISTS (
  SELECT 1 FROM UNNEST(experiences) AS e
  WHERE e.company_info.domain = 'example.com'
)
```

Testing one role with `e.company_info.domain != 'example.com'` does not exclude a matching company elsewhere in the person's history.

### Experience durations and company attributes

Find people with five to ten years of combined experience as a software engineer or software developer:

```sql
(
  SELECT SUM(r.duration_years)
  FROM UNNEST(experiences) AS c
  CROSS JOIN UNNEST(c.company_roles) AS r
  WHERE SEARCH(r.canonical_title, 'Software Engineer OR Software Developer')
) BETWEEN 5 AND 10
```

This sums the indexed durations of matching roles. Separate overlapping roles can contribute overlapping time. For overall experience, use `total_experience_years`. `SUM` requires a numeric field with scalar evaluation support, such as `duration_years`; it is not available for every numeric field. Null numeric values do not contribute.

Find people currently working at a SaaS company that is hiring and has 50 to 500 employees:

```sql
EXISTS (
  SELECT 1 FROM UNNEST(experiences) AS c
  CROSS JOIN UNNEST(c.company_roles) AS r
  WHERE r.end_date IS NULL
    AND c.company_info.is_saas = TRUE
    AND c.company_info.is_hiring = TRUE
    AND c.company_info.num_employees BETWEEN 50 AND 500
)
```

### Geography

Find people located in California, United States:

```sql
location_state_code = 'US-CA'
```

Use two-letter ISO country codes and country-prefixed subdivision codes for exact country and state filters. A subdivision code such as `US-CA` already identifies its country.

#### Named places and distance

Use `LOCATION('full place name', 'ISO country code')` for a city, state, or named area. For example, `LOCATION('Athens, Greece', 'GR')` filters for Athens in Greece. Preserve place qualifiers and use an uppercase two-letter country code; use `NULL` when the country is unspecified.

Find people located on the US West Coast:

```sql
LOCATION('West Coast, US', 'US')
```

Add a positive radius and a unit of `'km'` or `'mi'` for distance from the named place's center. For example, `LOCATION('New York, United States', 'US', 50, 'km')` searches within 50 km of New York.

Find people within 25 miles of the resolved center of the US West Coast region:

```sql
LOCATION('West Coast, US', 'US', 25, 'mi')
```

Combine place predicates with `AND`, `OR`, and `NOT`, alongside other profile conditions. Each place keeps its own country and radius. `LOCATION` applies to the person's location and belongs at the profile level, outside subqueries. Unknown places return HTTP 400.

#### Time zones

Use `location_timezone = 'label'` or `location_timezone IN ('label', ...)` for these supported time zones:

- `Eastern Time (US and Canada)`
- `Central Time (US and Canada)`
- `Mountain Time (US and Canada)`
- `Pacific Time (US and Canada)`
- `Atlantic Time (Canada)`
- `Newfoundland Time`
- `Alaska Time`
- `Hawaii Time`
- `India Standard Time`
- `China Standard Time`
- `Japan Standard Time`
- `Korea Standard Time`
- `Singapore Time`

Find people in either Pacific Time (US and Canada) or India Standard Time:

```sql
location_timezone IN ('Pacific Time (US and Canada)', 'India Standard Time')
```

`AND`, `OR`, and `NOT` can combine time-zone conditions. Unsupported labels or operations return HTTP 400.

#### Coordinates and regions

You can also filter numeric coordinates or use `SEARCH(location_regions, 'Europe OR APAC')` to match either indexed region label.

Find people inside the rectangle bounded by latitudes 37.0–38.5 and longitudes -123.0–-121.0:

```sql
location_coordinates.lat BETWEEN 37.0 AND 38.5
AND location_coordinates.lng BETWEEN -123.0 AND -121.0
```

Coordinate bounds define a rectangular area. Use the four-argument `LOCATION` form for a distance radius.

### Advanced scalar expressions

The restricted dialect also supports `CAST` and `SAFE_CAST` to `INT64`, `FLOAT64`, `STRING`, `BOOL`, and `TIMESTAMP`. Casts of constants are evaluated before index lookup. Casts of field values require scalar-expression support on the field. `CAST` fails on invalid values; `SAFE_CAST` produces a null value, which does not satisfy a comparison.

Supported correlated `MIN`/`SUM`, field-to-field comparisons, `IFNULL`/`COALESCE`, `DATE_ADD`/`DATE_SUB`, `TIMESTAMP_ADD`/`TIMESTAMP_SUB`, `EXTRACT(YEAR FROM ...)`, and numeric subtraction require scalar-expression support on every field read during that evaluation.

Find people whose job ended on or after their employer went public:

```sql
EXISTS (
  SELECT 1 FROM UNNEST(experiences) AS c
  CROSS JOIN UNNEST(c.company_roles) AS r
  WHERE r.end_date >= c.company_info.ipo_date
)
```

The tables distinguish scalar-capable fields from fields that only support ordinary indexed lookups. Unsupported expressions are rejected; SQL is not executed against arbitrary profile JSON.

## Complete field reference

Reference date: 2026-10-30. Every field supported by the endpoint's public SQL schema is listed below. Fields are optional; an absent value does not establish a negative fact about a person.

### Read the tables

- `string[]` denotes an array of scalar strings. Use `SEARCH(field, 'value')` for membership or alternatives.
- **Exact value** means whole-value equality, `IN`, or exact-value `SEARCH`. **Field text SEARCH** means term matching on this field; exact `=` comparisons are unsupported for tokenized fields.
- **Document text only** means the field contributes to `SEARCH(_text, ...)`, but field-specific `SEARCH(field, ...)` and exact comparisons are unsupported. Indexed presence tests remain available.
- **Numeric range**, **Date/time range**, and **Boolean** support compatible comparisons and presence tests.
- **Presence only** supports `IS NULL` and `IS NOT NULL`. **Scalar aggregate only** means a value can be read by supported scalar aggregate evaluation, not used as a direct indexed range predicate.
- **Scalar expressions** marks fields with the index's scalar postfilter capability, required for supported field-to-field comparisons, casts, date arithmetic, and aggregates. It does not make arbitrary SQL functions available.
- **Filter-only** fields can be used in filters but need not appear in returned profiles. Other fields can still be renamed, grouped, omitted, or redacted by the response formatter.
- Typed `timestamp` fields accept `DATE '2025-01-01'` or `TIMESTAMP '2025-01-01T00:00:00Z'`. A field typed `number` remains numeric even when its name contains `date` or `timestamp`; do not assume the types are interchangeable.

### Profile fields

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">has_<wbr>emails</span> | boolean<br />Boolean; scalar expressions | Indexed email-availability flag. Filtering it does not reveal email addresses. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">has_<wbr>phone_<wbr>numbers</span> | boolean<br />Boolean; scalar expressions | Indexed phone-availability flag. Filtering it does not reveal phone numbers. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">is_<wbr>student</span> | boolean<br />Boolean | Indexed student-status flag. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">canonical_<wbr>skills</span> | string[]<br />Exact value | Normalized canonical skill labels. [View canonical skills](https://gist.github.com/vslaykovsky/f8531a45ecad4bb57c28b772d3e10a39). |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">canonical_<wbr>title</span> | string[]<br />Exact value | Canonical title labels for the profile headline. [View canonical job titles](https://gist.github.com/vslaykovsky/33930db25c90204ea2dc708c33e7cc4c). |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">connections_<wbr>count</span> | number<br />Numeric range | Indexed LinkedIn connection count. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">estimated_<wbr>age</span> | number<br />Numeric range | Estimated person age in years; an inferred value that can be missing. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">first_<wbr>name</span> | string<br />Exact value | Given name. Matching normalizes case, whitespace, and Latin accents. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">followers_<wbr>count</span> | number<br />Numeric range | Indexed LinkedIn follower count. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">gender</span> | string<br />Exact value; scalar expressions | Indexed gender label when available. Values: `male`, `female`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">is_<wbr>decision_<wbr>maker</span> | number<br />Numeric range; scalar expressions | Numeric decision-maker indicator or score; this is not a boolean field. Values: `0`, `1`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">last_<wbr>activity_<wbr>timestamp</span> | number<br />Numeric range | Numeric timestamp of the most recent indexed LinkedIn activity. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">last_<wbr>name</span> | string<br />Exact value | Family name. Matching normalizes case, whitespace, and Latin accents. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">linkedin_<wbr>slug</span> | string<br />Exact value | LinkedIn profile identifier, corresponding to result docid. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location</span> | string<br />Presence only; IS NULL or IS NOT NULL | Human-readable profile location; use LOCATION for named-place filters. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>continent</span> | string<br />Exact value; scalar expressions | Indexed continent label. Values: `Africa`, `Antarctica`, `Asia`, `Europe`, `North America`, `Oceania`, `South America`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>coordinates.<wbr>lat</span> | number<br />Numeric range | Profile latitude in decimal degrees. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>coordinates.<wbr>lng</span> | number<br />Numeric range | Profile longitude in decimal degrees. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>country_<wbr>code</span> | string<br />Exact value; scalar expressions | Two-letter ISO country code, such as US or GB. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>state_<wbr>code</span> | string<br />Exact value; scalar expressions | Country-prefixed subdivision code, such as US-CA. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>regions</span> | string[]<br />Exact value; scalar expressions | Indexed geographic region labels, which can include overlapping regions. Values: `AMER`, `APAC`, `Africa`, `Americas`, `Antarctica`, `Asia`, `Australia and New Zealand`, `Caribbean`, `Central America`, `Central Asia`, `Channel Islands`, `EMEA`, `EU`, `Eastern Africa`, `Eastern Asia`, `Eastern Europe`, `Europe`, `Latin America and the Caribbean`, `Melanesia`, `Micronesia`, `Middle Africa`, `North America`, `Northern Africa`, `Northern America`, `Northern Europe`, `Oceania`, `Polynesia`, `South America`, `South-eastern Asia`, `Southern Africa`, `Southern Asia`, `Southern Europe`, `Sub-Saharan Africa`, `Western Africa`, `Western Asia`, `Western Europe`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">middle_<wbr>name</span> | string<br />Exact value | Middle name. Matching normalizes case, whitespace, and Latin accents. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">posted_<wbr>on_<wbr>linkedin_<wbr>timestamp</span> | number<br />Numeric range | Numeric timestamp of the most recent indexed original LinkedIn post. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">profile_<wbr>created_<wbr>date</span> | number<br />Numeric range | Numeric index representation of the LinkedIn account creation date. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">shared_<wbr>linkedin_<wbr>posts_<wbr>timestamp</span> | number<br />Numeric range | Numeric timestamp of the most recent indexed LinkedIn share. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">liked_<wbr>linkedin_<wbr>posts_<wbr>timestamp</span> | number<br />Numeric range | Numeric timestamp of the most recent indexed LinkedIn like. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">title</span> | string<br />Field text SEARCH | Profile headline or title, searched as text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">total_<wbr>experience_<wbr>years</span> | number<br />Numeric range | Indexed overall professional experience in years. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">skills</span> | string[]<br />Document text only; presence tests | Free-text skills listed on the profile. IS NOT NULL requires at least one skill; IS NULL matches profiles without skill values. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">location_<wbr>timezone</span> | string<br />Virtual time zone; = or IN | Profile location time zone; use a supported label with = or IN. Values: `Eastern Time (US and Canada)`, `Central Time (US and Canada)`, `Mountain Time (US and Canada)`, `Pacific Time (US and Canada)`, `Atlantic Time (Canada)`, `Newfoundland Time`, `Alaska Time`, `Hawaii Time`, `India Standard Time`, `China Standard Time`, `Japan Standard Time`, `Korea Standard Time`, `Singapore Time`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">summary</span> | string<br />Document text only; presence tests | Indexed professional summary text. Stemming enabled. |

### `alternativeSlugs` records

Each row belongs to one `alternativeSlugs` element. Apply related conditions inside the same `EXISTS` scope.

Find people whose LinkedIn profile is also known by the alias `example-slug`:

```sql
EXISTS (SELECT 1 FROM UNNEST(alternativeSlugs) AS s WHERE s.element = 'example-slug')
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">alternativeSlugs.<wbr>element</span> | string<br />Exact value | One alternative LinkedIn slug associated with the profile. |

### `activities` records

Each row belongs to one `activities` element. Apply related conditions inside the same `EXISTS` scope.

Find people whose LinkedIn activity mentions machine learning on or after January 1, 2025:

```sql
EXISTS (SELECT 1 FROM UNNEST(activities) AS a WHERE SEARCH(a.title, 'machine learning') AND a.activity_timestamp >= DATE '2025-01-01')
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">activities.<wbr>title</span> | string<br />Field text SEARCH | Title text for this record. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">activities.<wbr>action</span> | string<br />Exact value | Activity action. Standard labels: `Posted by` (authored post), `Shared by` (repost), `Liked by` (like). Other source values include personalized strings such as `<name> liked this`; this is not a closed enum. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">activities.<wbr>activity_<wbr>timestamp</span> | timestamp<br />Date/time range; scalar expressions | Typed timestamp for the activity; accepts DATE or TIMESTAMP literals. Filter-only; not stored as a profile value. |

### `recommendations` records

Each row belongs to one `recommendations` element. Apply related conditions inside the same `EXISTS` scope.

Find people with a recommendation mentioning technical leadership:

```sql
EXISTS (SELECT 1 FROM UNNEST(recommendations) AS r WHERE SEARCH(r.recommendation, 'technical leadership'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">recommendations.<wbr>recommendation</span> | string<br />Field text SEARCH | Recommendation text. Stemming enabled. |

### `certifications` records

Each row belongs to one `certifications` element. Apply related conditions inside the same `EXISTS` scope.

Find people with an AWS certification from Amazon:

```sql
EXISTS (SELECT 1 FROM UNNEST(certifications) AS c WHERE SEARCH(c.title, 'AWS') AND SEARCH(c.issuer, 'Amazon'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>begin_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed start date of the certification. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>begin_<wbr>timestamp</span> | number<br />Numeric range | Numeric start timestamp of the certification. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>end_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed end date; use IS NULL for a missing end date. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>end_<wbr>timestamp</span> | number<br />Numeric range | Numeric end timestamp; a missing value can represent an open-ended or undated record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>issuer</span> | string<br />Field text SEARCH | Certification issuer name, searched as text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">certifications.<wbr>title</span> | string<br />Field text SEARCH | Title text for this record. Stemming enabled. |

### `courses` records

Each row belongs to one `courses` element. Apply related conditions inside the same `EXISTS` scope.

Find people who list at least one course by name:

```sql
EXISTS (SELECT 1 FROM UNNEST(courses) AS c WHERE c.title IS NOT NULL)
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">courses.<wbr>organizer</span> | string<br />Document text only; presence tests | Course organizer, searched as text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">courses.<wbr>title</span> | string<br />Document text only; presence tests | Title text for this record. Stemming enabled. |

### `educations` records

Each row belongs to one `educations` element. Apply related conditions inside the same `EXISTS` scope.

Find people with a bachelor's, master's, or doctoral degree from Stanford:

```sql
EXISTS (SELECT 1 FROM UNNEST(educations) AS edu WHERE SEARCH(edu.degree, 'bachelor OR master OR doctor') AND SEARCH(edu.campus, 'Stanford'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>activities_<wbr>and_<wbr>societies</span> | string<br />Document text only; presence tests | Education activities and societies text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>campus</span> | string<br />Field text SEARCH | Institution or campus name, searched as text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>country</span> | string<br />Exact value | Country label attached to this institution or company; not necessarily a two-letter code. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>degree</span> | string[]<br />Exact value | Canonical degree labels. Values: `associate`, `bachelor`, `master`, `MBA`, `doctor`, `postdoc`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>end_<wbr>timestamp</span> | number<br />Numeric range | Numeric end timestamp; a missing value can represent an open-ended or undated record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>end_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed end date; use IS NULL for a missing end date. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>focus</span> | string<br />Exact value | Institution focus: FC = full comprehensive (five faculty areas and a medical school), CO = comprehensive (five areas), FO = focused (three or four areas), SP = specialist (two or fewer areas). Values: `FC`, `CO`, `FO`, `SP`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>is_<wbr>top_<wbr>local_<wbr>university</span> | boolean<br />Boolean | Indexed local top-university flag. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>is_<wbr>top_<wbr>university</span> | boolean<br />Boolean | Indexed top-university flag. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>major</span> | string<br />Document text only; presence tests | Major or subject text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>research</span> | string<br />Exact value | Institution research intensity: VH = very high, HI = high, MD = medium/moderate, LO = low. Values: `VH`, `HI`, `MD`, `LO`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>score</span> | number<br />Numeric range | Indexed numeric education or institution score. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>sequenceNo</span> | number<br />Numeric range | Sequence or order value from the source record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>size</span> | string<br />Exact value | Institution size: L = large, XL = extra large, M = medium, S = small. Values: `L`, `XL`, `M`, `S`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>specialization</span> | string<br />Document text only; presence tests | Education specialization text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>specialization_<wbr>category</span> | string<br />Exact value | Canonical education specialization-category label. Values: `Business & Management`, `Finance`, `Engineering`, `Computer Science & IT`, `Health & Medicine`, `Social Sciences`, `Natural Sciences`, `Mathematics`, `Education`, `Communication & Media`, `Arts, Design & Architecture`, `Humanities & Liberal Arts`, `Law & Criminal Justice`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>start_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed start date, usable with DATE or TIMESTAMP literals. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>start_<wbr>timestamp</span> | number<br />Numeric range | Numeric start timestamp. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">educations.<wbr>status</span> | string<br />Exact value | Indexed institution status, such as a public/private classification. Values: `Public`, `Private not for Profit`, `Private for Profit`. |

### `experiences` records

Each row belongs to one `experiences` element. Apply related conditions inside the same `EXISTS` scope.

This is the company-group scope. Access company attributes through `c.company_info`; join `c.company_roles AS r` using UNNEST for role fields. Role `location_info` belongs to the role rather than the company group. Use `r.end_date IS NULL` for current employment.

Find people with experience at a company with at least 100 employees:

```sql
EXISTS (SELECT 1 FROM UNNEST(experiences) AS c WHERE c.company_info.num_employees >= 100)
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>alternative_<wbr>domains</span> | string[]<br />Exact value | Alternative company domain values. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>annual_<wbr>revenue</span> | number<br />Numeric range | Indexed annual company revenue value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>business_<wbr>client_<wbr>sizes</span> | string[]<br />Exact value | Canonical company client-size labels. Values: `startups`, `SMBs`, `mid-market`, `enterprise`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>country</span> | string<br />Exact value | Country label attached to this institution or company; not necessarily a two-letter code. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>backed_<wbr>by_<wbr>y_<wbr>combinator</span> | boolean<br />Boolean | Whether the company is indexed as YC-backed. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>domain</span> | string<br />Exact value | Company domain, such as stripe.com. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>followers_<wbr>count</span> | number<br />Numeric range | Indexed company follower count. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>founded_<wbr>in</span> | number<br />Numeric range; scalar expressions | Company founding year as a number. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>funding_<wbr>total_<wbr>usd</span> | number<br />Numeric range | Total indexed company funding in USD. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>has_<wbr>ipo</span> | boolean<br />Boolean | Whether an IPO is recorded for the company. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>ipo_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed company IPO date. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>ai</span> | boolean<br />Boolean | Indexed AI-company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>b2b</span> | boolean<br />Boolean | Indexed business-to-business company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>b2c</span> | boolean<br />Boolean | Indexed business-to-consumer company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>cryptocurrency</span> | boolean<br />Boolean | Indexed cryptocurrency-company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>ecommerce</span> | boolean<br />Boolean | Indexed ecommerce-company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>hiring</span> | boolean<br />Boolean | Indexed company hiring flag. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>is_<wbr>saas</span> | boolean<br />Boolean | Indexed SaaS-company classification. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>industries</span> | string[]<br />Exact value | Canonical whole-value company industry labels. [View canonical industries](https://gist.github.com/vslaykovsky/7967f5a69c39fe4c958856ade32757fd). |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>keywords</span> | string[]<br />Document text only; presence tests | Company descriptive terms. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>last_<wbr>funding_<wbr>round_<wbr>date</span> | timestamp<br />Date/time range | Typed date of the latest indexed funding round. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>latest_<wbr>funding_<wbr>amount</span> | number<br />Numeric range | Amount of the latest indexed funding round. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>latest_<wbr>funding_<wbr>round</span> | string<br />Exact value | Whole-value label of the latest indexed funding round. Common labels: `Angel`, `Pre Seed`, `Seed`, `Series A`, `Series B`, `Series C`, `Series D`, `Series E`, `Series F`, `Series G`, `Series H`, `Series I`, `Series J`, `Other`, `Venture`, `Grant`. Other source labels can occur. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>linkedin_<wbr>slug</span> | string<br />Exact value | Company LinkedIn slug. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>name</span> | string<br />Field text SEARCH | Name text for the company or membership. No stemming. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>num_<wbr>employees</span> | number<br />Numeric range | Indexed company employee count. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>specialties</span> | string[]<br />Document text only; presence tests | Company specialties text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>technologies</span> | string[]<br />Document text only; presence tests | Company technology text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>type</span> | string<br />Exact value | Whole-value company type label. Values: `Privately Held`, `Public Company`, `Nonprofit`, `Government Agency`, `Partnership`, `Educational`, `Self-Owned`, `Self-Employed`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>valuation</span> | number<br />Numeric range | Indexed company valuation value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>website</span> | string<br />Exact value | Company website value; use domain for a domain-only match. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>yc_<wbr>batch</span> | string<br />Exact value | YC batch label, such as W22, when present. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>yc_<wbr>batch_<wbr>year</span> | number<br />Numeric range | YC batch year as a number. |

### `experiences.company_info.funding_rounds` records

Each row belongs to one `experiences.company_info.funding_rounds` element. Apply related conditions inside the same `EXISTS` scope.

Find people currently working at a company that raised a Series A round in 2024 or later:

```sql
EXISTS (SELECT 1 FROM UNNEST(experiences) AS c CROSS JOIN UNNEST(c.company_roles) AS r CROSS JOIN UNNEST(c.company_info.funding_rounds) AS fr WHERE r.end_date IS NULL AND fr.round_name = 'Series A' AND fr.date >= DATE '2024-01-01')
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>funding_<wbr>rounds.<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Date of the funding round or publication; typed versus numeric representation follows the Type column. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>funding_<wbr>rounds.<wbr>round_<wbr>name</span> | string<br />Exact value | Whole-value funding round label. Common labels: `Angel`, `Pre Seed`, `Seed`, `Series A`, `Series B`, `Series C`, `Series D`, `Series E`, `Series F`, `Series G`, `Series H`, `Series I`, `Series J`, `Other`, `Venture`, `Grant`. Other source labels can occur. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>info.<wbr>funding_<wbr>rounds.<wbr>value_<wbr>usd</span> | number<br />Scalar aggregate only | Funding-round amount in USD; available to supported scalar aggregates, not a direct indexed range lookup. Filter-only; not stored as a profile value. |

### `experiences.company_roles` records

Each row belongs to one `experiences.company_roles` element. Apply related conditions inside the same `EXISTS` scope.

Find people currently working as engineers:

```sql
EXISTS (SELECT 1 FROM UNNEST(experiences) AS c CROSS JOIN UNNEST(c.company_roles) AS r WHERE r.end_date IS NULL AND SEARCH(r.title, 'engineer'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>canonical_<wbr>title</span> | string[]<br />Exact value | Canonical whole-value title labels for this role. [View canonical job titles](https://gist.github.com/vslaykovsky/33930db25c90204ea2dc708c33e7cc4c). |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>company</span> | string<br />Field text SEARCH | Company name on the experience; tokenized without stemming. No stemming. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>is_<wbr>startup_<wbr>experience</span> | boolean<br />Boolean | Indexed startup-experience flag for this role. Values: `TRUE`, `FALSE`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>duration_<wbr>years</span> | number<br />Numeric range; scalar expressions | Indexed role duration in years; supports filtered SUM. Overlapping roles can overlap in the sum. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>end_<wbr>timestamp</span> | number<br />Numeric range | Numeric end timestamp; a missing value can represent an open-ended or undated record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>end_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed end date; use IS NULL for a missing end date. Filter-only; not stored as a profile value. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>experience_<wbr>summary</span> | string<br />Document text only; presence tests | Role description text, tokenized without stemming. No stemming. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>location_<wbr>info.<wbr>country_<wbr>code</span> | string<br />Exact value | Two-letter country code attached to this record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>location_<wbr>info.<wbr>lat</span> | number<br />Numeric range | Latitude in decimal degrees. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>location_<wbr>info.<wbr>lng</span> | number<br />Numeric range | Longitude in decimal degrees. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>location_<wbr>info.<wbr>location_<wbr>regions</span> | string[]<br />Exact value | Geographic region labels attached to this record. Values: `AMER`, `APAC`, `Africa`, `Americas`, `Antarctica`, `Asia`, `Australia and New Zealand`, `Caribbean`, `Central America`, `Central Asia`, `Channel Islands`, `EMEA`, `EU`, `Eastern Africa`, `Eastern Asia`, `Eastern Europe`, `Europe`, `Latin America and the Caribbean`, `Melanesia`, `Micronesia`, `Middle Africa`, `North America`, `Northern Africa`, `Northern America`, `Northern Europe`, `Oceania`, `Polynesia`, `South America`, `South-eastern Asia`, `Southern Africa`, `Southern Asia`, `Southern Europe`, `Sub-Saharan Africa`, `Western Africa`, `Western Asia`, `Western Europe`. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>sequenceNo</span> | number<br />Numeric range | Sequence or order value from the source record. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>start_<wbr>timestamp</span> | number<br />Numeric range | Numeric start timestamp. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>start_<wbr>date</span> | timestamp<br />Date/time range; scalar expressions | Typed start date, usable with DATE or TIMESTAMP literals. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">experiences.<wbr>company_<wbr>roles.<wbr>title</span> | string<br />Field text SEARCH | Title text for this record. Stemming enabled. |

### `languages` records

Each row belongs to one `languages` element. Apply related conditions inside the same `EXISTS` scope.

Find people who explicitly list English as a language:

```sql
EXISTS (SELECT 1 FROM UNNEST(languages) AS l WHERE l.language = 'English')
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">languages.<wbr>language</span> | string<br />Exact value | Whole-value language name. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">languages.<wbr>proficiency</span> | string<br />Exact value | Whole-value proficiency label, not a numeric level. Values: `A1`, `A2`, `B1`, `B2`, `C1`, `C2`. |

### `memberships` records

Each row belongs to one `memberships` element. Apply related conditions inside the same `EXISTS` scope.

Find people who list at least one membership by name:

```sql
EXISTS (SELECT 1 FROM UNNEST(memberships) AS m WHERE m.name IS NOT NULL)
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">memberships.<wbr>description</span> | string<br />Document text only; presence tests | Description text for this record; check the filtering column before using it in SEARCH. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">memberships.<wbr>name</span> | string<br />Document text only; presence tests | Name text for the company or membership. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">memberships.<wbr>title</span> | string<br />Document text only; presence tests | Title text for this record. Stemming enabled. |

### `patents` records

Each row belongs to one `patents` element. Apply related conditions inside the same `EXISTS` scope.

Find people with a patent about neural networks:

```sql
EXISTS (SELECT 1 FROM UNNEST(patents) AS p WHERE SEARCH(p.title, 'neural network'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">patents.<wbr>description</span> | string<br />Document text only; presence tests | Description text for this record; check the filtering column before using it in SEARCH. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">patents.<wbr>reference</span> | string<br />Document text only; presence tests | Patent reference text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">patents.<wbr>title</span> | string<br />Field text SEARCH | Title text for this record. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">patents.<wbr>url</span> | string<br />Exact value | URL attached to this patent or publication. |

### `publications` records

Each row belongs to one `publications` element. Apply related conditions inside the same `EXISTS` scope.

Find people with a publication about distributed systems:

```sql
EXISTS (SELECT 1 FROM UNNEST(publications) AS p WHERE SEARCH(p.title, 'distributed systems'))
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">publications.<wbr>description</span> | string<br />Document text only; presence tests | Description text for this record; check the filtering column before using it in SEARCH. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">publications.<wbr>issue</span> | string<br />Document text only; presence tests | Publication issue text. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">publications.<wbr>title</span> | string<br />Field text SEARCH | Title text for this record. Stemming enabled. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">publications.<wbr>url</span> | string<br />Exact value | URL attached to this patent or publication. |

### `inferred_languages` records

Each row belongs to one `inferred_languages` element. Apply related conditions inside the same `EXISTS` scope.

These languages are inferred; use `languages` when you require explicitly listed language evidence.

Find people inferred to speak English:

```sql
EXISTS (SELECT 1 FROM UNNEST(inferred_languages) AS l WHERE l.language = 'English')
```

| SQL path | Type and filtering | Meaning |
| --- | --- | --- |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">inferred_<wbr>languages.<wbr>language</span> | string<br />Exact value | Whole-value language name. |
| <span style="display: inline-block; min-width: 8rem; max-width: 11rem; font-family: monospace; overflow-wrap: anywhere;">inferred_<wbr>languages.<wbr>proficiency</span> | string<br />Exact value | Whole-value proficiency label, not a numeric level. Values: `A1`, `A2`, `B1`, `B2`, `C1`, `C2`. |
