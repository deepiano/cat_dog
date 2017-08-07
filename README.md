# cat_dog

1. Train model[0 - 8]

  $ python model0.py
  $ python model1.py
  $ python model2.py
  $ python model3.py
  $ python model4.py
  $ python model5.py
  $ python model6.py
  $ python model7.py
  $ python model8.py
  
2. Make csv submission file.

  $ python ensemble.py
  $ cd result
  $ sort -k1 -n -t, result.csv >> submission.csv
