from sklearn.utils import all_estimators
from inspect import signature, _empty

estimators = all_estimators(type_filter='regressor')

for i, (name, obj) in enumerate(estimators):
    params = [val[1] for val in signature(obj).parameters.items()]
    all_optional = True
    for param in params:
        if param.default == _empty:
            all_optional = False

    if not all_optional:
        print(f"{name} has mandatory params!!! Oh No!!!")


