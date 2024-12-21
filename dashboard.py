import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

st.set_page_config(page_title="My SFT",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: My Student Feedback on Teaching")

file_name="All_SFT.xlsx"

# Load data
def load_data(file_name):
    df = pd.read_excel(file_name, sheet_name="Data")
    SFT_courses = pd.read_excel(file_name, sheet_name="Courses")
    SFT_criteria = pd.read_excel(file_name, sheet_name="Criteria")
    return df, SFT_courses, SFT_criteria

df, SFT_courses, SFT_criteria = load_data(file_name)

# Function for filtering data
def filter_data(df, years, subject_domain, course_level):
    return df[
        (df["AcadYear"].isin(years)) &
        (df["Discipline"].isin(subject_domain)) &
        (df["Stage"].isin(course_level))
    ]

# Function for calculating metrics
def calculate_metrics(filtered_df, filtered_SFT_courses):
    unique_courses = filtered_SFT_courses["CourseCode"].nunique()
    total_tutorials = filtered_SFT_courses["NumTutorials"].sum()
    total_students = filtered_SFT_courses["NumStudents"].sum()
    Total_tutorials = total_tutorials * 13

    unique_courses_df = filtered_df.drop_duplicates(subset=["CourseCode"])
    stage_counts = unique_courses_df["Stage"].value_counts()
    foundation_count = stage_counts.get("Foundation", 0)
    intermediate_count = stage_counts.get("Intermediate", 0)
    advanced_count = stage_counts.get("Advanced", 0)

    return {
        "unique_courses": unique_courses,
        "total_tutorials": total_tutorials,
        "Total_tutorials": Total_tutorials,
        "total_students": total_students,
        "foundation_count": foundation_count,
        "intermediate_count": intermediate_count,
        "advanced_count": advanced_count
    }

# Function for calculating average scores
def calculate_average_scores(filtered_df, score_col, duration):
    if score_col == "Score":
        filtered_df["TotalResponses"] = (
            filtered_df["Strongly Agree"] +
            filtered_df["Agree"] +
            filtered_df["Neutral"] +
            filtered_df["Disagree"] +
            filtered_df["Strongly Disagree"])
    elif score_col == "Favor":
        filtered_df["TotalResponses"] = (
            filtered_df["Strongly Agree"] +
            filtered_df["Agree"])
        
    if duration == "Overall":
        filtered_df["WeightedScore"] = filtered_df[score_col] * filtered_df["TotalResponses"]
        overall_totals = filtered_df.agg({"WeightedScore": "sum", "TotalResponses": "sum"})
        average_score = overall_totals["WeightedScore"] / overall_totals["TotalResponses"]
    elif duration == "AcadYearSem":
        filtered_df["WeightedScore"] = filtered_df[score_col] * filtered_df["TotalResponses"]
        semester_totals = (filtered_df.groupby(["AcadYear", "Semester"], as_index=False).agg({"WeightedScore": "sum", "TotalResponses": "sum"}))
        semester_totals["AverageRating"] = semester_totals["WeightedScore"] / semester_totals["TotalResponses"]
        average_score = semester_totals

    return average_score

def calculate_criteria_aggregates(filtered_df, min_year, max_year, criteria_column="Criteria", score_column="Score", favor_column="Favor"):
    # Filter by selected years
    filtered_df = filtered_df[(filtered_df["AcadYear"].astype(int) >= min_year) & (filtered_df["AcadYear"].astype(int) <= max_year)]

    # Calculate average scores and favorability by criteria
    average_scores = filtered_df.groupby(["AcadYear", criteria_column])[score_column].mean().unstack()
    favor_scores = filtered_df.groupby(["AcadYear", criteria_column])[favor_column].mean().unstack()

    return average_scores, favor_scores

def sort_tutorials_within_domain(aggregated_data):
    """Sort tutorials within each domain so earlier years come first."""
    tutorial_years = {grp: int(grp.split('(')[-1].strip('.0)')) for grp in aggregated_data.columns}
    sorted_columns = sorted(aggregated_data.columns, key=lambda x: (tutorial_years[x], x))
    return aggregated_data[sorted_columns]

def calculate_course_aggregates(filtered_df, selected_course, group_by_columns, score_column="Score", favor_column="Favor"):
    filtered_df = filtered_df[(filtered_df["CourseCode"] + " - " + filtered_df["CourseName"] == selected_course)]

    # Grouping data
    aggregated_scores = filtered_df.groupby(group_by_columns)[score_column].mean().unstack()
    aggregated_favor = filtered_df.groupby(group_by_columns)[favor_column].mean().unstack()

    # Sort by mean values
    aggr_scores_mean = filtered_df.groupby(["Criteria"])[score_column].mean().sort_values(ascending=False).index
    aggregated_scores = aggregated_scores.loc[aggr_scores_mean]
    aggr_favor_mean = filtered_df.groupby(["Criteria"])[favor_column].mean().sort_values(ascending=False).index
    aggregated_favor = aggregated_favor.loc[aggr_favor_mean]

    return aggregated_scores, aggregated_favor

def plot_slope(data,min_year,max_year,color_palette):  
    # Prepare data for plotting
    fig = go.Figure()

    for i, domain in enumerate(data.columns):
        
        # Get values for min and max years
        min_value = data.loc[min_year, domain] if min_year in data.index else None
        max_value = data.loc[max_year, domain] if max_year in data.index else None

        # Determine line style (solid for increase, dashed for decrease)
        line_style = "solid" if pd.notna(min_value) and pd.notna(max_value) and max_value >= min_value else "dash"
        
        # Use the custom color palette for the domain
        color = color_palette.get(domain, "#333333")

        fig.add_trace(
            go.Scatter(
                x=[0,1],
                y=data.loc[[min_year, max_year], domain].values,
                mode="lines+markers",
                name=domain,
                marker=dict(size=10),
                line=dict(width=2, dash=line_style, color=color),
            )
        )

        # Add annotations only for non-NaN values
        if not pd.isna(min_value):
            # Min Year Annotation (Right-Aligned)
            fig.add_annotation(
                x=0-0.02,
                y=min_value,
                text=f"{domain} ({min_value:.2f})",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor="right",
            )
        if not pd.isna(max_value):
            # Max Year Annotation (Left-Aligned)
            fig.add_annotation(
                x=1+0.02,
                y=max_value,
                text=f"{domain} ({max_value:.2f})",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor="left",
            )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(
            tickvals=[0,1],
            ticktext=[str(min_year), str(max_year)],
            title=f"Rating Changes between {min_year} and {max_year}"),
        yaxis=dict(title="Criteria Rating"),
        legend_title="Domains",
    )

    # Add a vertical line for the min_year
    fig.add_shape(
        type="line",
        x0=0, 
        x1=0,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dot",
        ),
        layer="below",
    )

    # Add a vertical line for the max_year
    fig.add_shape(
        type="line",
        x0=1,
        x1=1,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dot",
        ),
        layer="below",
    )
    return fig

def yearly_plot_slope(data, min_year, max_year, color_palette):
    fig = go.Figure()

    # Filter data for the selected range of years
    available_years = [year for year in range(min_year, max_year + 1) if year in data.index]

    for i, domain in enumerate(data.columns):
        # Extract values for all available years
        yearly_values = data.loc[available_years, domain].dropna()
        
        # Use the custom color palette for the domain
        color = color_palette.get(domain, "#333333")

        # Plot lines and markers for each domain
        for j in range(len(yearly_values.index) - 1):
            current_year = yearly_values.index[j]
            next_year = yearly_values.index[j + 1]
            
            if current_year == 2018 and next_year == 2023:
                line_style = "dot"
            else:
                current_value = yearly_values[current_year]
                next_value = yearly_values[next_year]
                line_style = "solid" if next_value >= current_value else "dash"

            fig.add_trace(
                go.Scatter(
                    x=[current_year, next_year],
                    y=[yearly_values[current_year], yearly_values[next_year]],
                    mode="lines+markers",
                    name=domain,
                    marker=dict(size=10),
                    line=dict(width=2, dash=line_style, color=color),
                    showlegend=(j == 0)
                )
            )

        # Add annotations only for min and max years
        if min_year in yearly_values.index:
            min_value = yearly_values[min_year]
            fig.add_annotation(
                x=min_year-0.2,
                y=min_value,
                text=f"{domain} ({min_value:.2f})",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor="right",
            )
        if max_year in yearly_values.index:
            max_value = yearly_values[max_year]
            fig.add_annotation(
                x=max_year+0.2,
                y=max_value,
                text=f"{domain} ({max_value:.2f})",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor="left",
            )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(
            tickvals=available_years,
            title="Years",
        ),
        yaxis=dict(title="Criteria Rating"),
        legend_title="Domains",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title_font=dict(size=12),
            font=dict(size=10),
            itemwidth=30,
            traceorder="normal",
            valign="middle",
        ),
    )

    def add_line(year):
        # Add vertical lines for the min_year and max_year
        fig.add_shape(
            type="line",
            x0=year,
            x1=year,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color="gray",
                width=1,
                dash="dot",
            ),
            layer="below",
        )
    
    for year in available_years:
        add_line(year)

    return fig

def plot_bar(aggregated_data, aggregated_stdev,color_palette):
    indices = np.arange(len(aggregated_data))
    fig = go.Figure()

    # Plot data
    for i, tutorial in enumerate(aggregated_data.columns):
        fig.add_trace(go.Bar(
            x=indices + i * 0.1,  # Bar positions
            y=aggregated_data[tutorial],
            name=tutorial,
            marker=dict(color=color_palette[i % len(colors)]),
            width=0.1
        ))

    # Annotate mean values
    for idx, (mean_value, stdev_value) in enumerate(zip(aggregated_data.mean(axis=1), aggregated_stdev)):
        fig.add_annotation(
            x=indices[idx],
            y=5.1,
            text=f"{mean_value:.2f} ± {stdev_value:.2f}",
            xanchor="left",
            showarrow=False,
            font=dict(size=12, color="black")
        )

    # Update layout
    fig.update_layout(
        barmode="group",
        xaxis=dict(
            title="Criteria",
            tickvals=indices,
            ticktext=aggregated_data.index,
            tickangle=45
        ),
        yaxis=dict(title="Criteria Rating", range=[3.9, 5.1], dtick=0.1, minor=dict(ticks='outside', tick0=4, dtick=0.05)),
        legend_title="Tutorial Group",
    )

    return fig


###### MAIN SCRIPT #####
with st.sidebar:
    st.title("Dashboard parameters")
    st.markdown("Use the controls in this sidebar to change the range and type of data displayed in the dashboard.")
    st.markdown("---")

    year_range = st.slider("**Teaching Years:**", 2016, 2024, [2016, 2024])
    years = np.arange(year_range[0], year_range[1] + 1)
    st.markdown("Graduate TAs at SUTD were not rated by students. No teaching ratings for 2019-2022.")
    st.markdown("---")

    subjects = ["Physics", "Mathematics", "Interdisciplinary"]
    subject_domain = st.multiselect("**Subject Domains:**", options=subjects, default=subjects)
    st.markdown("---")

    levels = ["Foundation", "Intermediate", "Advanced", "Graduate"]
    undergraduate = ["Foundation", "Intermediate", "Advanced"]
    course_level = st.multiselect("**Course Level:**", options=levels, default=undergraduate)

    filtered_df = filter_data(df, years, subject_domain, course_level)
    filtered_SFT_courses = filter_data(SFT_courses, years, subject_domain, course_level)

tab_names = ["**Summary**", "**Detailed Reports**", "**Criteria Description**"]
tabs = st.tabs(tab_names)

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] p{
		gap: 2px;
        font-size: 35px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #ffe8e8;
		border-radius: 8px 8px 0px 0px;
		gap: 1px;
		padding: 10px;
        
    }

	.stTabs [aria-selected="true"] {
  		background-color: #ff2b2b;
        color: white;
        padding: 10px;
	}

</style>""", unsafe_allow_html=True)

with tabs[0]:
    # Summary Tab
    metrics = calculate_metrics(filtered_df, filtered_SFT_courses)
    overall_average_rating = calculate_average_scores(filtered_df,"Score","Overall")
    overall_favor_rating = calculate_average_scores(filtered_df,"Favor","Overall")
    
    col1_1, col1_2, col1_3, col1_4, col1_5 = st.columns((1,1,1,1,1))
    
    with col1_1:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{metrics["unique_courses"]}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Unique Courses Taught</div>", unsafe_allow_html=True)
        st.markdown(f"""
                    <div style='text-align: center; font-size:14px;'>
                        <strong>{metrics["foundation_count"]}</strong> Foundation, 
                        <strong>{metrics["intermediate_count"]}</strong> Intermediate, and 
                        <strong>{metrics["advanced_count"]}</strong> Advanced level courses.
                    </div>
                    """,
                    unsafe_allow_html=True)
    with col1_2:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{metrics["total_tutorials"]}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Classes Instructed</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{metrics["Total_tutorials"]}</strong> classes in lecture and tutorial formats. </div>", unsafe_allow_html=True)
    with col1_3:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{metrics["total_students"]}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Students Taught</div>", unsafe_allow_html=True)
        if "Graduate" in course_level:
            st.markdown(f"<div style='text-align: center; font-size:14px;'> both undergraduate and graduate students. </div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: center; font-size:14px;'> from freshmen to final-year students. </div>", unsafe_allow_html=True)
    with col1_4:
        overall_average_rating_percent = overall_average_rating/5*100
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{overall_average_rating_percent:.1f}%</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Average SFT Rating</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{overall_average_rating:.2f}</strong> out of 5 </div>", unsafe_allow_html=True)
    with col1_5:
        overall_favor_rating_percent = overall_favor_rating/5*100
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{overall_favor_rating_percent:.1f}%</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Average Favorability</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{overall_favor_rating:.2f}</strong> out of 5 </div>", unsafe_allow_html=True)

   
    col2_1, col2_2 = st.columns((1,3))

    with col2_1:
        select_metric = st.selectbox("",options=["SFT Score", "Favorability"])
        st.markdown("**SFT Score:** The average score attained by the faculty based on all student feedback ratings collected.")
        st.markdown("**Favorability:** The average score attained by the faculty based on all _positive_ student feedback ratings collected.")
        st.markdown("\* The overall average favorability data is unavailable.")
    with col2_2:
        if select_metric == "SFT Score":
            semester_totals = calculate_average_scores(filtered_df,"Score","AcadYearSem")
        elif select_metric == "Favorability":
            semester_totals = calculate_average_scores(filtered_df,"Favor","AcadYearSem")
        semester_totals["YearSemester"] = semester_totals["AcadYear"].astype(str).str.replace("\.0", "", regex=True) + " " + semester_totals["Semester"].astype(str)

        ### BAR CHART SETTINGS ###
        def assign_color(year):
            if year == 2016:
                return "Undergraduate TA"
            elif 2017 <= year <= 2018:
                return "Project Officer (TEL)"
            elif 2023 <= year <= 2024:
                return "Part-time Instructor"
            else:
                return "Others"
        
        semester_totals["ColorCategory"] = semester_totals["AcadYear"].apply(assign_color)

        color_map = {
            "Undergraduate TA": "lightblue",
            "Project Officer (TEL)": "lightgreen",
            "Part-time Instructor": "lightcoral",
            "Others": "gray",
        }

        fig = px.bar(
            semester_totals,
            x="YearSemester",
            y="AverageRating",
            color="ColorCategory",  # Use the color category for custom colors
            color_discrete_map=color_map,  # Apply the custom color map
            labels={"YearSemester": "Year and Semester", "AverageRating": "Average Rating", "ColorCategory": "Teaching Role"},
        )
        fig.update_yaxes(range=[4, 5])

        df2 = pd.read_excel(file_name,sheet_name="Overall")

        # Ensure df2 corresponds only to x-axis values available in the bar chart
        df2_filtered = pd.merge(
            semester_totals["YearSemester"],
            df2,
            on="YearSemester",
            how="inner"
        )

        # Add a scatter plot for the overall averages
        fig.add_scatter(
            x=df2_filtered["YearSemester"],  # Use filtered x-axis values
            y=df2_filtered["Overall"],
            mode="markers",  # Line with markers
            name="Overall Average Score",
            marker=dict(size=10, symbol="diamond", color='red'),
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig,use_container_width=True)
    
with tabs[1]:
    st.subheader("Criteria Report")
    
    # Criteria Report Tab
    # Multi-select for filtering
    unique_criteria = SFT_criteria["Criteria Shortname"].dropna().unique()
    default_criteria = ["Encouraged Engagement", "Communicated Clearly", "Supported Understanding", "Nurtured Critical Thinking", "Demonstrated Approachability"]
    selected_criteria = st.multiselect("**Select Criterion:**", options=unique_criteria, default=default_criteria)

    min_year, max_year = year_range

    # Filter data for criteria analysis
    filtered_criteria_df = filter_data(filtered_df, years, subject_domain, course_level)
    filtered_criteria_df = filtered_criteria_df[filtered_criteria_df["Criteria"].isin(selected_criteria)]

    # Calculate aggregates
    average_scores, favor_scores = calculate_criteria_aggregates(filtered_criteria_df, min_year, max_year)

    # Generate a color palette for criteria
    num_criteria = len(unique_criteria)
    generated_colors = pc.qualitative.Dark24[:num_criteria]
    dynamic_color_palette = {criteria: generated_colors[i] for i, criteria in enumerate(unique_criteria)}

    # Check if data is available for the selected min_year and max_year
    if min_year not in average_scores.index or max_year not in average_scores.index:
        st.warning("Data not available for the selected start/end year.")
    else:
        fig1 = yearly_plot_slope(average_scores,year_range[0],year_range[1],dynamic_color_palette)
        fig2 = yearly_plot_slope(favor_scores,year_range[0],year_range[1],dynamic_color_palette)
    
        with st.expander("**SFT Rating Evolution for Selected Criterion**"):
            st.plotly_chart(fig1, use_container_width=True)
        with st.expander("**Favorability Rating Evolution for Selected Criterion**"):
            st.plotly_chart(fig2, use_container_width=True)   

    st.subheader("Course Report")
    # Sorting and filtering courses
    filtered_courses = filter_data(SFT_courses, years, subject_domain, course_level)
    filtered_courses = filtered_courses.sort_values(by="CourseCode")
    selected_course = st.selectbox("**Select a Course:**", (filtered_courses["CourseCode"] + " - " + filtered_courses["CourseName"]).unique())

    # Combine TutorialGrp and AcadYear into a unique identifier
    df["TutorialGrp_Year"] = df["TutorialGrp"] + " (" + df["AcadYear"].astype(str).str.replace("\.0", "", regex=True) + ")"

    # Filter df based on the selected course
    filtered_df = df[(df["CourseCode"] + " - " + df["CourseName"] == selected_course)]

    # Aggregate data for the selected course
    aggregated_scores, aggregated_favor = calculate_course_aggregates(
        df, selected_course, group_by_columns=["Criteria", "TutorialGrp_Year"]
    )
    
    # Sort tutorials within each domain
    aggregated_scores = sort_tutorials_within_domain(aggregated_scores)
    aggregated_favor = sort_tutorials_within_domain(aggregated_favor)
    
    # Standard deviations for annotations
    score_stdev = df.groupby(["Criteria"])["ScoreStdev"].mean()
    favor_stdev = df.groupby(["Criteria"])["FavStdev"].mean()

    # Prepare indices and colors
    indices = np.arange(len(aggregated_scores))
    unique_tutorials = aggregated_scores.columns
    colors = px.colors.qualitative.Pastel

    fig3 = plot_bar(aggregated_scores, score_stdev,colors)
    fig4 = plot_bar(aggregated_favor, favor_stdev,colors)

    with st.expander("**SFT Rating by Tutorial for Selected Course**"):
        st.plotly_chart(fig3, use_container_width=True)
    with st.expander("**Favorability Rating by Tutorial for Selected Course**"):
        st.plotly_chart(fig4, use_container_width=True)

with tabs[2]:
    st.table(SFT_criteria)
