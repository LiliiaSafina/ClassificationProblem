import pandas as pd
import preprocessing


preprocessing.write_to_file(preprocessing.get_data_frame("data\\adult.data"), "data\\df_train.scv")
preprocessing.write_to_file(preprocessing.get_data_frame("data\\adult.test"), "data\\df_test.scv")
