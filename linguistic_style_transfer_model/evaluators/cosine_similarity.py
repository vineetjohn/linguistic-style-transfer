from scipy.spatial.distance import cosine


class CosineSimilarity:
    def __init__(self, actual_content):
        self.actual_content = actual_content

    def get_similarity(self, style_and_content):
        style, content = style_and_content
        cosine_sim = 1 - cosine(content, self.actual_content)
        return cosine_sim, style
