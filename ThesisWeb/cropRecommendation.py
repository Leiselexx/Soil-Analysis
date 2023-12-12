import pandas as pd


data = pd.read_excel('cropsData.xlsx')
df = pd.DataFrame(data)

class RecommendCrops:


    def recommend_crops_for_pH(self, pH_level, df):
        pH_level = round(pH_level, 1)
        recommended_crops = df[df['pHLevel'] == pH_level]['Crops'].unique()

        recommended_crops_text = "\n".join(
            recommended_crops) if recommended_crops.size > 0 else 'No recommended crops found'

        return recommended_crops_text

