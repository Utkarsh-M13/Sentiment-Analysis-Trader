PY=python

stage1:
	$(PY) -m nst.train.train_stage1

stage2:
	$(PY) -m nst.train.train_stage2_spy
