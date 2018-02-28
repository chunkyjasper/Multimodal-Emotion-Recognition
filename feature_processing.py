from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import config
# take each arousal/valance csv, take average of the result and output a new csv

output_path = config.DATA_DIR / "RECOLA-Annotation-avg"

def convert(subject_id, type="arousal", format="%.2f"):
    if type != "arousal" and type != "valence" :
        exit(1)
    path = config.DATA_DIR / "RECOLA-Annotation" / "emotional_behaviour" / type / "P{}.csv".format(subject_id)
    csv = np.loadtxt(str(path), delimiter=';', skiprows=1)
    new_csv = np.rot90(np.vstack((np.mean(csv[:, 1:], axis=1), csv[:,0])), 3)
    np.savetxt(str(output_path / type / "P{}.csv".format(subject_id)), new_csv, delimiter=",", fmt=format)

if __name__ == "__main__":
  for subject_id in config.SUBJECT_LIST:
      convert(subject_id, type="arousal")

