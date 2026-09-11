import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib


train = pd.read_csv('data/train.csv')
print('Data has been loaded')

X_train = train.drop('income', axis=1)
y_train = train['income']


gb_model = GradientBoostingClassifier(random_state=42, learning_rate=0.2, max_depth=3, n_estimators=200)

gb_cv = cross_val_score(gb_model, X_train, y_train, cv=5)
gb_model.fit(X_train, y_train)
print('Training process completed successfully')


joblib.dump(gb_model, 'final_model_weights.pkl')