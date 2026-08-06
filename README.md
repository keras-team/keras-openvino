# OpenVino backend implementation for Keras

## Local development

For development, you need the `keras` repository and the `keras-openvino`
repository checked out locally. That's because the unit tests code is in the
keras repository.

We first check out the main `keras` repository and the `pluggable_backend`
branch. Then, we install the required dependencies.

```
gh repo clone keras-team/keras
cd keras
git checkout pluggable_backend
pip install -r requirements.txt
cd ..
```

Assuming you have a fork of `keras-openvino`, you will run the following. This
also installs `keras-openvino` locally so that `keras` can find and import the
`keras-openvino` module.

```
gh repo clone <your_github_handle>/keras-openvino
cd keras-openvino
pip install -r requirements.txt
pip install -e .
cd ..
```

Running tests happens from the root of the `keras` repository.

```
cd keras
KERAS_BACKEND=openvino pytest keras --ignore=keras/src/applications
```
