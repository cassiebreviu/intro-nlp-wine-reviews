#%%
import logging
import azure.functions as func
from azure.storage.blob import BlockBlobService
import numpy as np
import pandas as pd
from joblib import dump, load
#%%

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
                
        block_blob_service = BlockBlobService(account_name=ACCTNAME, account_key=ACCKEY)
        block_blob_service.get_blob_to_path('winemodels',MODELNAME,MODELNAME)
        block_blob_service.get_blob_to_path('winemodels',VECTNAME,VECTNAME)
        trained_lr = load_file(MODELNAME)
        vectorizer = load_file(VECTNAME)

        #print(trained_lr)
        #print(vectorizer)
#%%
        #name = "fruity awesomeness of all proptions. I like wine it tastes good."
        x = vectorizer.transform(np.array([name]))
        #result = trained_lr.predict(x)
    
        proba = trained_lr.predict_proba(x)
        classes = trained_lr.classes_
        df = pd.DataFrame(data=proba, columns=classes)
        topPrediction = df.T.sort_values(by=[0], ascending = [False])
        #print(topPrediction)
        return(topPrediction.to_json(orient='index'))
#%%
        #return func.HttpResponse(f"Hello {description}!")
        #return func.HttpResponse(f"Hello {name}!")
    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )
#%%
def load_file(fileName):
    with open(fileName, 'rb') as file_model:
        return load(file_model)

#%%
