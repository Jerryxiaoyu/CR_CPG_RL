


import os
import shutil
import glob
from utils import *
import csv
import pandas as pd

class G:
	output_dir = None
	output_file = None
	first_row = True
	log_headers = []
	log_current_row = {}

def configure_log_dir(logname, name, copy = True):
	"""
	Set output directory to d, or to /tmp/somerandomnumber if d is None
	"""

	path = os.path.join(log_dir(), logname, name)
	os.makedirs(path)  # create path
	if copy:
		filenames = glob.glob('*.py')  # put copy of all python files in log_dir
		for filename in filenames:  # for reference
			shutil.copy(filename, path)
	return path

def log_tabular(key, val):
	"""
	Log a value of some diagnostic
	Call this once for each diagnostic quantity, each iteration

	examples:
			logz.log_tabular('Iteration', i_iter)
			logz.log_tabular('AverageCost', log['avg_reward'])
			logz.log_tabular('MinimumCost', log['min_reward'])
			logz.log_tabular('MaximumCost', log['max_reward'])
			logz.dump_tabular()
	"""
	if G.first_row:
		G.log_headers.append(key)
	else:
		assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
	assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
	G.log_current_row[key] = val


def dump_tabular():
	"""
	Write all of the diagnostics from the current iteration
	"""
	vals = []
	key_lens = [len(key) for key in G.log_headers]
	max_key_len = max(15,max(key_lens))
	keystr = '%'+'%d'%max_key_len
	fmt = "| " + keystr + "s | %15s |"
	n_slashes = 22 + max_key_len
	print("-"*n_slashes)
	for key in G.log_headers:
		val = G.log_current_row.get(key, "")
		if hasattr(val, "__float__"): valstr = "%8.3g"%val
		else: valstr = val
		print(fmt%(key, valstr))
		vals.append(val)
	print("-"*n_slashes)
	if G.output_file is not None:
		if G.first_row:
			G.output_file.write("\t".join(G.log_headers))
			G.output_file.write("\n")
		G.output_file.write("\t".join(map(str,vals)))
		G.output_file.write("\n")
		G.output_file.flush()
	G.log_current_row.clear()
	G.first_row=False


class LoggerCsv(object):
	""" Simple training logger: saves to file and optionally prints to stdout
	V1.0 可以实时存储
	"""
	
	def __init__(self, logdir, csvname='log'):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""
		self.path = os.path.join(logdir, csvname + '.csv')
		self.write_header = True
		self.log_entry = {}
		self.f = open(self.path, 'w')
		self.writer = None  # DictWriter created with first call to write() method
		self.open_flag = True
	
	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)
		
		if self.open_flag:
			if self.write_header:
				fieldnames = [x for x in self.log_entry.keys()]
				self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
				self.writer.writeheader()
				self.write_header = False
		
		else:
			self.f = open(self.path, 'a+')
			self.open_flag = True
			fieldnames = [x for x in self.log_entry.keys()]
			self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
		
		self.writer.writerow(self.log_entry)
		self.log_enbtry = {}
		self.close()
	
	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		'''
		print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
															   log['_MeanReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				print('{:s}: {:.3g}'.format(key, log[key]))
		'''
		#print('log writed!')
		#print('\n')
	
	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)
	
	def close(self):
		""" Close log file - log cannot be written after this """
		
		self.open_flag = False
		self.f.close()
	
	def log_table2csv(self, data, header=True):
		df = pd.DataFrame(data)
		df.to_csv(self.path, index=False, header=header)
	
	def log_csv2table(self):
		data = pd.read_csv(self.path, header=0, encoding='utf-8')
		return np.array(data)