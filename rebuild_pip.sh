sudo pip uninstall -y tensorflow-serving-api
bazel build -c opt --config=cuda //tensorflow_serving/tools/pip_package:build_pip_package
bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package /tmp/tensorflow_serving_package
mv /tmp/tensorflow_serving_package/tensorflow_serving_api-undefined-py2-none-any.whl /tmp/tensorflow_serving_package/tensorflow_serving-1.2.1-cp27-cp27mu-linux_x86_64.whl
sudo pip install /tmp/tensorflow_serving_package/tensorflow_serving-1.2.1-cp27-cp27mu-linux_x86_64.whl
