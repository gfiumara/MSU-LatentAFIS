Building for Modern Systems
---------------------------

 1. Install system packages

    * Ubuntu 20.04.03 LTS:
       ```sh
       sudo apt install python2 virtualenv python-tk libsm6 gcc g++ python2-dev libeigen3-dev libboost-filesystem-dev
       ```

    * CentOS 8.2
       ```sh
       sudo dnf install python2 python2-virtualenv python2-tkinter libSM python2-devel eigen3-devel gcc-g++
       ```

 2. Configure a virtual Python 2 environment
    ```sh
    virtualenv -p /usr/bin/python2.7 msu_env
    source msu_env/bin/activate
    ```

 3. Use `pip` to install Python dependencies.
    ```sh
    pip install tensorflow==1.3.0 torch==0.4.0 torchvision==0.2.1 scipy==1.1.0 scikit-learn==0.20.1 scikit-image==0.14.2 keras-preprocessing==1.0.1 opencv-python==3.4.2.17 tensorpack==0.8.9 psutil==5.4.8
    ```
    *  **Note**: The version of `numpy` and `scikit-image` differ slightly than what is in environment.yaml. The version of `numpy` is required by the version of `tensorflow` specified, so it is assumed this was a typo. When using the newer version of `numpy`, a newer patch of `scikit-image` is required. No deviations were observed from sample data scores with these two updates.
    *  **Note**: If offline, download the appropriate wheel or source files from pypi directly and then `pip install *`. You must install all at once or dependencies will not be able to be resolved.
       * https://pypi.org/project/backports.functools-lru-cache/1.6.4/#files
       * https://pypi.org/project/backports.weakref/1.0.post1/#files
       * https://pypi.org/project/bleach/1.5.0/#files
       * https://pypi.org/project/cloudpickle/1.3.0/#files
       * https://pypi.org/project/cycler/0.10.0/#files
       * https://pypi.org/project/dask/1.2.2/#files
       * https://pypi.org/project/decorator/4.4.2/#files
       * https://pypi.org/project/enum34/1.1.10/#files
       * https://pypi.org/project/funcsigs/1.0.2/#files
       * https://pypi.org/project/functools32/3.2.3.post2/#files
       * https://pypi.org/project/futures/3.3.0/#files
       * https://pypi.org/project/html5lib/0.9999999/#files
       * https://pypi.org/project/keras/2.7.0/#files
       * https://pypi.org/project/Keras-Preprocessing/1.0.1/#files
       * https://pypi.org/project/kiwisolver/1.1.0/#files
       * https://pypi.org/project/Markdown/3.1.1/#files
       * https://pypi.org/project/matplotlib/2.2.5/#files
       * https://pypi.org/project/mock/3.0.5/#files
       * https://pypi.org/project/msgpack/1.0.3/#files
       * https://pypi.org/project/msgpack-numpy/0.4.7.1/#files
       * https://pypi.org/project/networkx/2.2/#files
       * https://pypi.org/project/numpy/1.16.6/#files
       * https://pypi.org/project/opencv-python/3.4.2.17/#files
       * https://pypi.org/project/Pillow/6.2.2/#files
       * https://pypi.org/project/protobuf/3.17.3/#files
       * https://pypi.org/project/psutil/5.4.8/#files
       * https://pypi.org/project/pyarrow/0.16.0/#files
       * https://pypi.org/project/pyparsing/2.4.7/#files
       * https://pypi.org/project/python-dateutil/2.8.2/#files
       * https://pypi.org/project/pytz/2021.3/#files
       * https://pypi.org/project/PyWavelets/1.0.3/#files
       * https://pypi.org/project/pyzmq/19.0.2/#files
       * https://pypi.org/project/scikit-image/0.14.2/#files
       * https://pypi.org/project/scikit-learn/0.20.1/#files
       * https://pypi.org/project/scipy/1.1.0/#files
       * https://pypi.org/project/six/1.16.0/#files
       * https://pypi.org/project/subprocess32/3.5.4/#files
       * https://pypi.org/project/tabulate/0.8.9/#files
       * https://pypi.org/project/tensorflow/1.3.0/#files
       * https://pypi.org/project/tensorflow-tensorboard/0.1.8/#files
       * https://pypi.org/project/tensorpack/0.8.9/#files
       * https://pypi.org/project/termcolor/1.1.0/#files
       * https://pypi.org/project/toolz/0.10.0/#files
       * https://pypi.org/project/torch/0.4.0/#files
       * https://pypi.org/project/torchvision/0.2.1/#files
       * https://pypi.org/project/tqdm/4.62.3/#files
       * https://pypi.org/project/Werkzeug/1.0.1/#files

 4. Build the comparison algorithm.
    ```
    make -C matcher
    ```

    **Note**: Use the `brl` branch for code changes.

Notes
-----
 * The `models` directory hierarchy contains files that start with `._`. Delete all of these.
   ```sh
   find models -name '\._*' -delete
   ```
 * `--idir` and `--tdir` arguments **require** trailing slash.
 * Exemplar and latents should be separated in input and output.
 * Template generation requires ~16 GB of RAM to operate.
