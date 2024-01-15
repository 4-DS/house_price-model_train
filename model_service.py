from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact, TextFileArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model'),
            PickleArtifact('test_data'),
            TextFileArtifact('service_version',
                             file_extension='.txt',
                             encoding='utf8')]) # for versions of bentoml 0.13 and newer   
class ModelService(BentoService): 
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df.values)
    
    @api(input=JsonInput(), batch=False)
    def test_data(self, *args): 
        """ Return some test data for running a test """
        return self.artifacts.test_data
    