# Passage model development

**Development data only; no authorship validation claim.**

| Language | Words | Model | All-author accuracy | Matched-genre accuracy | Unknown accepted | Known accepted |
|---|---:|---|---:|---:|---:|---:|
| grc | 500 | function_centroid | 73.1% | 72.6% | 0.0% | 0.0% |
| grc | 500 | morph_centroid | 75.9% | 65.9% | 0.0% | 0.0% |
| grc | 500 | char_linear | 73.2% | 64.9% | 0.0% | 0.0% |
| grc | 1000 | function_centroid | 79.4% | 61.9% | 0.0% | 0.0% |
| grc | 1000 | morph_centroid | 75.0% | 52.8% | 0.0% | 0.0% |
| grc | 1000 | char_linear | 73.6% | 57.1% | 0.0% | 0.0% |
| grc | 2000 | function_centroid | 76.6% | 58.6% | 0.0% | 0.0% |
| grc | 2000 | morph_centroid | 74.5% | 52.4% | 0.0% | 0.0% |
| grc | 2000 | char_linear | 75.6% | 52.0% | 0.0% | 0.0% |
| hbo | 500 | function_centroid | 62.2% | 62.2% | Unavailable | Unavailable |
| hbo | 500 | morph_centroid | 65.4% | 65.4% | Unavailable | Unavailable |
| hbo | 500 | char_linear | 75.3% | 75.3% | Unavailable | Unavailable |
| hbo | 1000 | function_centroid | 67.0% | 67.0% | Unavailable | Unavailable |
| hbo | 1000 | morph_centroid | 61.5% | 61.5% | Unavailable | Unavailable |
| hbo | 1000 | char_linear | 61.0% | 61.0% | Unavailable | Unavailable |
| hbo | 2000 | function_centroid | 69.1% | 69.1% | Unavailable | Unavailable |
| hbo | 2000 | morph_centroid | 56.8% | 56.8% | Unavailable | Unavailable |
| hbo | 2000 | char_linear | 43.8% | 43.8% | Unavailable | Unavailable |

Selected designs (locked before fresh evaluation):

- grc: function_centroid, 500 tokens; development operating targets met = False.
- hbo: char_linear, 500 tokens; development operating targets met = False.

Selection uses matched-genre accuracy after preferring feasible rejection. Zero false acceptance with zero known acceptance is not success. Function-word order and suffixes are structural proxies, not validated syntactic or morphological annotation. All rates are author/work balanced. Detailed matched-topic controls and independent work partitions are in `development.json`.
