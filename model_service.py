from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import TextFileArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model'),
            TextFileArtifact('service_version',
                             file_extension='.txt',
                             encoding='utf8')]) # for versions of bentoml 0.13 and newer   
class ModelService(BentoService): 
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df.values) 

    @api(input=JsonInput(), batch=False)
    def service_version(self, *args): 
        """ Return version of a running service """
        return self.artifacts.service_version