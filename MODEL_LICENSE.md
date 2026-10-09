# Model weights licence

**The source code** in this repository is Apache License 2.0 (`LICENSE`).

**The trained model weights** released for LayaMacro are licensed under
**[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.

## Why the weights are non-commercial

The released models were trained on three datasets (see `NOTICE`):

| dataset | licence |
|---|---|
| Nutrition5k | CC BY 4.0 |
| DFood-TJ-F-Nutr | CC0 1.0 |
| NutritionVerse-Real | **CC BY-NC-SA 4.0** |

Whether trained weights are "adapted material" of their training data is legally unsettled.
We take the cautious reading: weights trained on NutritionVerse-Real inherit its
non-commercial, share-alike terms. You may use, share and adapt the weights for
**non-commercial** purposes, with attribution, and must share adaptations under the same
licence.

The weights contain only what we trained (the projector, the trained parts of the decision
model and any fine-tuned encoder layers). The base models (Laya, SigLIP 2) are downloaded
from their official sources under their own Apache-2.0 licences.

## Not a medical device

LayaMacro estimates nutrition from a photo with a probability range. It is a research
artifact: it is not a medical device, is not validated for clinical or dietary-management
use, and must not be used to make health decisions. Errors of 40-60 kcal per meal are
typical on the data it was trained on, and much larger on unfamiliar kinds of photos.
