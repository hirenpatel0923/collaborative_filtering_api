class DataParams():
    def __init__(self):
        self.BASE_DIR = './'
        self.DATA_DIR = 'data/'
        self.MOVIE_CSV = self.BASE_DIR + self.DATA_DIR + 'movie.csv'
        self.RATINGS_CSV = self.BASE_DIR + self.DATA_DIR + 'rating.csv'
        self.LINK_CSV = self.BASE_DIR + self.DATA_DIR + 'link.csv'
        self.USERIDS_CSV = self.BASE_DIR + self.DATA_DIR + 'userIds.csv'