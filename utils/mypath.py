class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10'}
        assert(database in db_names)

        if database == 'cifar-10':
            return 'data/gruntdata/dataset'

        else:
            raise NotImplementedError
