import pandas as pd
from datetime import datetime

#Load Dataset

df = pd.read_csv("netflix_titles.csv")
print("✔ Dataset loaded successfully\n")


#Schema Validation

print("Initial Data Types:\n")
print(df.dtypes, "\n")

# date_added → datetime
if df['date_added'].dtype != 'datetime64[ns]':
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    print("'date_added' converted to datetime\n")
else:
    print("'date_added' already datetime\n")

# release_year → int
if df['release_year'].dtype != 'int64':
    df['release_year'] = df['release_year'].astype('int64')
    print("'release_year' converted to integer\n")
else:
    print("'release_year' already integer\n")

print("Schema after validation:\n")
df.info()
print("\n")


#Logical Consistency Check
#(release_year ≤ year_added)

df['added_year'] = df['date_added'].dt.year

invalid_years = df[df['release_year'] > df['added_year']]

if len(invalid_years) > 0:
    print(f"⚠ Found {len(invalid_years)} inconsistent records")
    print(invalid_years[['title', 'release_year', 'added_year']].head(), "\n")
    df = df[df['release_year'] <= df['added_year']]
    print("Invalid records removed\n")
else:
    print("No logical inconsistencies found\n")


#Duplicate Detection

exact_duplicates = df.duplicated().sum()
print(f"Exact duplicate rows found: {exact_duplicates}\n")

duplicate_titles = df[df.duplicated(subset=['title'], keep=False)]

if not duplicate_titles.empty:
    print("Duplicate titles across regions detected")
    print(
        duplicate_titles[['title', 'country', 'type']]
        .sort_values('title')
        .head(10),
        "\n"
    )
else:
    print("No duplicate titles found\n")

print("Most frequently repeated titles:\n")
print(df['title'].value_counts().head(10), "\n")

before = len(df)
df = df.drop_duplicates(subset=['title', 'type', 'country'])
after = len(df)
print(f"Removed {before - after} redundant duplicate records\n")


#Missing Value Handling

df = df.fillna("Unknown")

print("Missing values after cleaning:\n")
print(df.isnull().sum(), "\n")

#FEATURE ENGINEERING

current_year = datetime.now().year

# 1. Content age
df['content_age'] = current_year - df['release_year']

# 2. Time to Netflix
df['time_to_netflix'] = df['added_year'] - df['release_year']

# 3. Genre count
df['genre_count'] = df['listed_in'].apply(
    lambda x: len(x.split(', ')) if x != "Unknown" else 0
)

# 4. Cast count
df['cast_count'] = df['cast'].apply(
    lambda x: len(x.split(', ')) if x != "Unknown" else 0
)

# 5. Kids content flag
kids_ratings = ['G', 'PG', 'TV-G', 'TV-Y', 'TV-Y7']
df['is_kids_content'] = df['rating'].apply(
    lambda x: 1 if x in kids_ratings else 0
)

# 6. Continent mapping
continent_map = {
    'United States': 'North America',
    'Canada': 'North America',
    'India': 'Asia',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'China': 'Asia',
    'United Kingdom': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',
    'Spain': 'Europe',
    'Brazil': 'South America',
    'Mexico': 'North America',
    'Australia': 'Oceania'
}

def map_continent(country):
    if country == "Unknown":
        return "Unknown"
    return continent_map.get(country.split(',')[0], "Other")

df['continent'] = df['country'].apply(map_continent)

# 7. Pandemic content flag
df['pandemic_content'] = df['release_year'].apply(
    lambda x: 1 if 2019 <= x <= 2021 else 0
)


print("Engineered features preview:\n")
print(
    df[
        [
            'content_age',
            'time_to_netflix',
            'genre_count',
            'cast_count',
            'is_kids_content',
            'continent',
            'pandemic_content'
        ]
    ].head()
)

print("\nFinal dataset description:\n")
print(df.describe(include='all'))


output_file = "netflix_titles_updated.xlsx"

df.to_excel(output_file, index=False)

print(f"✔ Updated dataset successfully saved as '{output_file}'")
