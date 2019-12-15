import matplotlib.pyplot as plt
import pandas as pd
# Regression chart, we will see more of this chart in the next class.
def chart_regression(pred, y):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    t.sort_values(by=['y'], inplace=True)
    _ = plt.plot(t['y'].tolist(), label='expected')
    _ = plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()