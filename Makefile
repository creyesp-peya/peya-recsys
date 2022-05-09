upload:
	#python setup.py sdist
	python setup.py sdist && gsutil cp ./dist/recsys-0.1.tar.gz gs://daf-vertex-poc-stg/recsys/dist/recsys-0.1.tar.gz