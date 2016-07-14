import shutil
origin_dir='/Users/hanzhang/Google Drive/data/dogvscat'

for i in xrange(1400,12500):
    shutil.move(origin_dir+'/large_dataset/cat.'+str(i)+'.jpg',origin_dir+'/train/cat')
    shutil.move(origin_dir+'/large_dataset/dog.'+str(i)+'.jpg',origin_dir+'/train/dog')

'''
for i in xrange(1000,1400):
    shutil.move(origin_dir+'/train1/cat.'+str(i)+'.jpg',origin_dir+'/cat/validate')
    shutil.move(origin_dir+'/train1/dog.'+str(i)+'.jpg',origin_dir+'/dog/validate')
'''
