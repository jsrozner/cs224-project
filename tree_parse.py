from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

archive = load_archive("./rep_allennlp/elmo-constituency-parser-2018.03.14.tar.gz" )
predictor = Predictor.from_archive(archive, 'constituency-parser')

predictor.predict_json({"sentence": "This is a sentence to be predicted!"})