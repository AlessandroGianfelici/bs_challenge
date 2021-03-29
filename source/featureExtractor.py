import pandas as pd
import numpy as np
import logging
from pampy import match, _
from functools import reduce, partial
from operator import add
import holidays
logger = logging.getLogger("__main__")

class FeatureExtraxtor(object):

    def __init__(self, path):
        self._data = pd.read_csv(path, parse_dates=['install_timestamp',	'free_trial_timestamp'])
        self._perimetro = self._data['uid'].drop_duplicates()
        return

    @property
    def data(self):
        return self._data

    def getFeatures(self):
        """
        This function merges all the features toghether and fill the missing values.
        The choice to compute all the features group in different function and to put 
        all togheter at the end has the advantage of making easy to add/remove new features
        from the etl, but is inefficient from a performance point of view.
        I've chosen this method because of the dimension of the dataset, with a larger dataset
        the best choice would be to perform all the operation in place on the original dataset
        """ 
        # N.B. I will pass self.data instead of self._data for safety reason (it is readonly)
        features = self.mergeFeatures([
            self._perimetro,
            self.getCountry(self.data),
            self.getDelay(self.data),
            self.getDevice(self.data),
            self.getHoliday(self.data),
            self.getGender(self.data),
            self.getAge(self.data),
            self.getAttributionNetwork(self.data),
            self.getLanguage(self.data),
            self.othersNumericalFeatures(self.data)
        ]).set_index('uid')

        for col in features.columns:
            features[col] = features[col].fillna((features[col].mean()))

        # I will delete now some irrelevant features (they have importances=0 according to the feature importances graph)
        features = features.drop(columns=['is_from_Oceania','language_other','is_from_Africa',
                                          'product_free_trial_length','iPod'], errors='ignore')
        return features.reset_index()

    @staticmethod
    def getLanguage(data):
        """
        This function apply the one hot encoding method to the language column
        """
        logger.info("*** getting language features...")
        # I am considering only the most frequent languages
        encodingFunction = lambda x : match(x, 
                                            'en', 'english',
                                            'fr', 'french',
                                            'de', 'german',
                                            'es', 'spanish',
                                             _, 'other')
        data['language_clean'] = data['language'].str.lower().apply(encodingFunction)
        return pd.get_dummies(data[['uid', 'language_clean']].set_index('uid'), prefix='language').reset_index()

    @staticmethod
    def getAge(data):
        """
        This function returns the age of the customer at the installation time.
        """
        logger.info("*** getting age features...")
        data['age'] = data['install_timestamp'].dt.year - data['onboarding_birth_year']
        # According to Apple's Terms and Conditions for the iTunes Store, the minimum age requirement for use of the service is 13 years old.
        # So I assume that the datapoint with age < 13 are invalid data.
        # I am also assuming that there aren't customer older than 99 (even if, in principle, older customers could exist, I think it's more likely
        # that those are invalid data).
        return data.loc[(data['age']>=13) & (data['age']<=99)][['uid', 'age']]
        
    @staticmethod
    def othersNumericalFeatures(data):
        """
        This function returns the columns of the input dataframe that can be used by the model without any proprocessing.
        """
        logger.info("*** getting numerical features...")

        return data[['uid',
                    'product_price_tier',
                    'product_periodicity',
                    'product_free_trial_length',
                    'net_purchases_15d']]

    @staticmethod
    def getGender(data):
        """
        This function generate gender related feature as a simple encoding 1 : Male, -1 : Female.
        All missing data are treated as 0.
        """
        logger.info("*** getting gender features...")

        # I am defining the encoding function. I am using the pampy package because it simplify the 
        # substitution of invalid values
        encodingFunction = lambda x : match(x, 
                                            'M', 1,
                                            'F', -1,
                                             _, 0)
        data['gender'] = data['onboarding_gender'].str.upper().apply(encodingFunction)
        return data[['uid','gender']]

    def getCountry(self, data):
        """
        This function extract the features related to the country, using one hot encoding.
        I noticed that the first 6 category take into account 75% of the total cases, so I
        will declare a column for each one and I will encode the other countries using their
        continent.
        """
        logger.info("*** getting geographical features...")

        # I will download a full list of country and continent codes from datahub.io
        countryList = pd.read_csv(r"https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv")
        # In my EDA I found the following to be the most common country in our dataset, so I will consider just these
        encodedCountries = ['US', 'GB', 'FR', 'CA', 'DE', 'AU']
        encodedContinents = ['Asia', 'Europe', 'Africa', 'Oceania', 'North America', 'South America']

        # I am creating a dataset with both country and continent code
        mergedDF = pd.merge(data[['uid', 'country']], 
                            countryList[['Two_Letter_Country_Code', 'Continent_Name']],
                            left_on = 'country', right_on='Two_Letter_Country_Code')

        # I will add a prefix to make our feature names more understandable
        addPrefix = partial(add, 'is_from_')
        featuresList = map(addPrefix, encodedCountries + encodedContinents)

        # I am applying the mapping rule
        getCountryOrContinent = partial(self.countryOrContinent, encodedCountries=set(encodedCountries))
        mergedDF['is_from'] = mergedDF.apply(getCountryOrContinent, axis=1)

        # Finally I am applying the one hot encoding
        return pd.get_dummies(mergedDF[['uid', 'is_from']].set_index('uid'), prefix='is_from')[featuresList].reset_index()

    @staticmethod
    def getDelay(data):
        """
        This function returns the delta, expressed in days, between 
        the installation of the app and the subscription.
        """
        logger.info("*** getting delay features...")

        data['Time_between_install_and_subs'] = (pd.to_datetime(data['free_trial_timestamp']) - pd.to_datetime(data['install_timestamp'])).dt.days
        # I assumed that only the data points with a free_trial_timestamp subsequent to install_timestamp are valid,
        # so I discard the others. N.B. this excludes also the cases where one of the two variables is missing.
        return data.loc[data['Time_between_install_and_subs']>=0][['uid', 'Time_between_install_and_subs']]

    def getHoliday(self, data):
        """
        This function returns a binary features which is 1 if the user started the free trial on holiday, 0 elsewhere
        """
        logger.info("*** getting calendar day features...")
        data['is_holiday'] = data.apply(self.isHolidays, axis=1)
        return data[['uid', 'is_holiday']]

    @staticmethod
    def countryOrContinent(df, encodedCountries : set):
        """
        This function take the dataframe and a list of countries to be considered. 
        If a country included in the list is given, this function returns its name,
        if a different country is given it returns the continent name.
        """
        if encodedCountries.intersection({df['country']}):
            return df['country']
        else:
            return df['Continent_Name']

    def getAttributionNetwork(self, data):
        """
        This function return a one hot encoding of the attribution network
        """
        logger.info("*** getting attribution network features...")

        # I am considering only the valid network from train dataset
        validNetworks = ['Snapchat Installs', 'Organic', 'Facebook Installs',
                         'Adwords UAC Installs', 'Apple Search Ads']
        # I will add a prefix to make our feature names more understandable
        addPrefix = partial(add, 'attribution_network_')
        featuresList = map(addPrefix, validNetworks)
        return pd.get_dummies(data[['uid', 'attribution_network']].set_index('uid'), prefix = 'attribution_network')[featuresList].reset_index()

    def getDevice(self, data):
        """
        This function return a one hot encoding of the device type (iPhone, iPod or iPad)
        """
        logger.info("*** getting device type features...")

        return self.generateFeatures(data, 'device_type', ['iPhone', 'iPod', 'iPad'])[['uid', 'iPhone', 'iPod', 'iPad']]

    def aggregate(self, data, col: str, feature: str):
        """
        This function aggregate different categorical features into a single binary variable.
        It uses partial string matching to find and aggregate similar product (I mean, 
        aggregate into a single binary flag all iPhone or all iPad)
        """
        df = data.copy()
        df[feature] = df[col].str.contains(feature, case=False).astype(float).fillna(0).astype(int)
        return df

    def generateFeatures(self, df, col: str, features: list):
        """
        The scope of this function is to apply iteratively the self.aggregate method and store the
        results (that are columns of 1 and 0 values) in new columns of the dataframe. In this way, we 
        obtain a result similar to the one hot encoding but based on partial (instead of exact) string matching.
        """
        return self.aggregate(df, col, features[0]) if (len(features) == 1) else self.generateFeatures(self.aggregate(df, col, features[-1]), col, features[0:-1])

    @staticmethod
    def isHolidays(df):
        """
        This function returns 1 if the free trial has been activated on holiday, 0 elsewhere.
        """
        try:
            # I am using the holidays package. Not every country is supported, so if a country is missing, I simply consider justweekend as holiday.
            holidaysInTheYear = holidays.CountryHoliday(df['country'], years = df['install_timestamp'].year)
            is_holiday = (df['install_timestamp'] in holidaysInTheYear)
        except:
            is_holiday = False
        return int(is_holiday or (df['install_timestamp'].weekday() > 4))

    def mergeFeatures(self, data_frames):
        """
        This method take as input a list of dataframes and return their outer join on the column 'uid'.
        """
        return reduce(lambda left, right: pd.merge(left,right,on='uid', how='outer'), data_frames)