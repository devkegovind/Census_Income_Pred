import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception Occured in Prediction")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 workclass:str,
                 education:str,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 sex:str,
                 native_country:str,
                 age:int,
                 education_num:int,
                 hours_per_week:int):
            self.workclass=workclass
            self.education=education
            self.marital_status=marital_status
            self.occupation= occupation
            self.relationship=relationship
            self.race=race
            self.sex=sex
            self.native_country=native_country
            self.age=age
            self.education_num=education_num
            self.hours_per_week=hours_per_week
    
    def get_data_as_dataframe(self):
         try:
              custom_data_input_dict = {
                'workclass':[self.workclass],
                'education':[self.education],
                 'marital-status':[self.marital_status],
                 'occupation':[self.occupation],
                 'relationship':[self.relationship],
                 'race':[self.race],
                 'sex':[self.sex],
                 'native-country':[self.native_country],
                 'age':[self.age],
                 'education-num':[self.education_num],
                 'hours-per-week':[self.hours_per_week]
              }

              df = pd.DataFrame(custom_data_input_dict)
              logging.info("Dataframe Gathered")
              return df
         except Exception as e:
              logging.info("Exception Occcured in prediction pipeline")
              raise CustomException(e, sys)
              


'''import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception Occurred in Prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 workclass:str,
                 education:str,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 sex:str,
                 native_country:str,
                 age:int,
                 education_num:int,
                 hours_per_week:int):
            self.workclass=workclass
            self.education=education
            self.marital_status=marital_status
            self.occupation= occupation
            self.relationship=relationship
            self.race=race
            self.sex=sex
            self.native_country=native_country
            self.age=age
            self.education_num=education_num
            self.hours_per_week=hours_per_week
    
    def get_data_as_dataframe(self):
         try:
              custom_data_input_dict = {
                'workclass':[self.workclass],
                'education':[self.education],
                 'marital_status':[self.marital_status],
                 'occupation':[self.occupation],
                 'relationship':[self.relationship],
                 'race':[self.race],
                 'sex':[self.sex],
                 'native_country':[self.native_country],
                 'age':[self.age],
                 'education_num':[self.education_num],
                 'hours_per_week':[self.hours_per_week]
              }

              df = pd.DataFrame(custom_data_input_dict)
              logging.info("Dataframe Gathered")
              return df
         except Exception as e:
              logging.info("Exception Occurred in prediction pipeline")
              raise CustomException(e, sys)'''



