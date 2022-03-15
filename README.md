## Performance

### T5

|Model                   |BoolQ|CB         |COPA|MultiRC    |ReCoRD     |RTE |WiC |WSC |
|:-:                     |:-:  |:-:        |:-: |:-:        |:-:        |:-: |:-: |:-: |
|Metric                  |Acc  |AvgF1 / Acc|Acc |F1a  / EM  |F1   / Acc |Acc |Acc |Acc |
|T5-11b (official_test)  |91.2 |93.9 / 96.8|94.8|88.1 / 63.3|94.1 / 93.4|92.5|76.9|93.8|
|T5-11b (bmtrain_val)    |90.1 |93.2 / 96.4|97.0|88.0 /     |           |92.1|77.2|    |
|GPT2-base (bmtrain_val) |     |           |    |           |           |    |    |    |