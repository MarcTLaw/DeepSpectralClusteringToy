import numpy
from tempfile import TemporaryFile

nb_training_examples = 15000
nb_test_examples = 6000

nb_classes = 3

save_directory = 'saved_data'

a = numpy.fromfile('%s/X_train.txt' % save_directory,sep=" ")
b = numpy.reshape(a,[nb_training_examples,nb_classes])
outfile = open('%s/X_train.npy' % save_directory,'wb')
numpy.save(outfile, b)
outfile.close()

a = numpy.fromfile('%s/X_test.txt' % save_directory,sep=" ")
b = numpy.reshape(a,[nb_test_examples,nb_classes])
outfile = open('%s/X_test.npy' % save_directory,'wb')
numpy.save(outfile, b)
outfile.close()



a = numpy.fromfile('%s/Y_train.txt' % save_directory,sep=" ")
b = numpy.reshape(a,[nb_training_examples,nb_classes])
outfile = open('%s/Y_train.npy' % save_directory,'wb')
numpy.save(outfile, b)
outfile.close()

a = numpy.fromfile('%s/Y_test.txt' % save_directory,sep=" ")
b = numpy.reshape(a,[nb_test_examples,nb_classes])
outfile = open('%s/Y_test.npy' % save_directory,'wb')
numpy.save(outfile, b)
outfile.close()


