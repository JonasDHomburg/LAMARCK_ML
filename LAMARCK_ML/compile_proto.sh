cd ..
protoc -I=./ --python_out=./ ./LAMARCK_ML/data_util/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/datasets/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/architectures/variables/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/architectures/functions/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/architectures/losses/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/architectures/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/individuals/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/individuals/implementations/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/reproduction/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/models/*.proto
protoc -I=./ --python_out=./ ./LAMARCK_ML/utils/*.proto