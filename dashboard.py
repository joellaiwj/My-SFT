import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

st.set_page_config(page_title="My SFT",page_icon=":bar_chart:",layout="wide")

st.title(":bar_chart: My Student Feedback on Teaching")

file_name="All_SFT.xlsx"
df = pd.read_excel(file_name,sheet_name="Data")
SFT_courses = pd.read_excel(file_name,sheet_name="Courses")
SFT_criteria = pd.read_excel(file_name,sheet_name="Criteria")

# Font size settings
font_settings = dict(
    family="Arial, sans-serif",
    size=16
)

tab_names = ["**Summary**", "**Detailed Reports**", "**Criteria Description**"]
tabs = st.tabs(tab_names)

with st.sidebar:
    st.title("Dashboard parameters")
    st.markdown("Use the controls in this sidebar to change the range and type of data displayed in the dashboard.")
    st.markdown("---")

    year_range = st.slider(":date: **Teaching Years:**",min_value=2016,max_value=2024,value=[2016,2024],key="year_range")
    st.markdown("I do not have teaching ratings for 2019-2022 as Graduate TAs were not rated by students in SUTD.")
    years = np.int64(np.linspace(year_range[0],year_range[1],year_range[1]-year_range[0]+1))
    years = np.array(years)
    st.markdown("---")

    subjects = ["Physics","Mathematics","Interdisciplinary"]
    subject_domain = st.multiselect(":school: **Subject Criteria:**",options=subjects,default=subjects)
    st.markdown("---")

    levels = ["Foundation","Intermediate","Advanced"]
    course_level = st.multiselect(":books: **Course Level:**",options=levels,default=levels)
    st.markdown("---")

    # Filtering the data
    filtered_df = df[
        (df["AcadYear"].isin(years)) &
        (df["Discipline"].isin(subject_domain)) &
        (df["Stage"].isin(course_level))
    ]

    # Filtering the courses
    filtered_SFT_courses = SFT_courses[
        (SFT_courses["AcadYear"].isin(years)) &
        (SFT_courses["Discipline"].isin(subject_domain)) &
        (SFT_courses["Stage"].isin(course_level))
    ]


with tabs[2]:
    st.table(SFT_criteria)

with tabs[0]:
    # Metrics calculations
    unique_courses = filtered_SFT_courses["CourseCode"].nunique()  # Count of unique CourseCode
    total_tutorials = filtered_SFT_courses["NumTutorials"].sum()  # Sum of NumTutorials
    total_students = filtered_SFT_courses["NumStudents"].sum()  # Sum of NumStudents

    Total_tutorials = total_tutorials*13

    # Get unique courses
    unique_courses_df = filtered_df.drop_duplicates(subset=["CourseCode"])  
    
    # Count occurrences of Foundation, Intermediate, and Advanced in Stage
    stage_counts = unique_courses_df["Stage"].value_counts()
    foundation_count = stage_counts.get("Foundation", 0)
    intermediate_count = stage_counts.get("Intermediate", 0)
    advanced_count = stage_counts.get("Advanced", 0)

    st.header("Summary")
    col1_1, col1_2, col1_3, col1_4, col1_5 = st.columns((1,1,1,1,1))
    
    with col1_1:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{unique_courses}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Unique Courses Taught</div>", unsafe_allow_html=True)
        st.markdown(f"""
                    <div style='text-align: center; font-size:14px;'>
                        <strong>{foundation_count}</strong> Foundation, 
                        <strong>{intermediate_count}</strong> Intermediate, and 
                        <strong>{advanced_count}</strong> Advanced level courses.
                    </div>
                    """,
                    unsafe_allow_html=True)
    with col1_2:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{total_tutorials}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Classes Instructed</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{Total_tutorials}</strong> classes in lecture and tutorial formats. </div>", unsafe_allow_html=True)
    with col1_3:
        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{total_students}</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Students Taught</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> from freshmen to final-year students. </div>", unsafe_allow_html=True)
    with col1_4:
        ## Step 1: Filter by the selected years
        filtered_by_years = filtered_df[filtered_df["AcadYear"].isin(years)]

        # Step 2: Sum the columns for feedback
        filtered_by_years["TotalResponses"] = (
            filtered_by_years["Strongly Agree"]
            + filtered_by_years["Agree"]
            + filtered_by_years["Neutral"]
            + filtered_by_years["Disagree"]
            + filtered_by_years["Strongly Disagree"]
        )

        # Step 3: Calculate the weighted score
        filtered_by_years["WeightedScore"] = filtered_by_years["Score"] * filtered_by_years["TotalResponses"]

        # Step 4: Aggregate to find overall totals
        overall_totals = filtered_by_years.agg({"WeightedScore": "sum", "TotalResponses": "sum"})

        # Step 5: Calculate the overall average rating
        overall_average_rating = overall_totals["WeightedScore"] / overall_totals["TotalResponses"]
        overall_average_rating_percent = overall_average_rating/5*100

        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{overall_average_rating_percent:.1f}%</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Average SFT Score</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{overall_average_rating:.2f}</strong> out of 5 </div>", unsafe_allow_html=True)

    
    with col1_5:
        # Step 2: Sum the columns for feedback (Strongly Agree and Agree only)
        filtered_by_years["TotalPositiveResponses"] = (
            filtered_by_years["Strongly Agree"] + filtered_by_years["Agree"]
        )

        # Step 3: Calculate the weighted score
        filtered_by_years["WeightedPositiveScore"] = filtered_by_years["Score"] * filtered_by_years["TotalPositiveResponses"]

        # Step 4: Aggregate to find overall totals
        overall_totals_positive = filtered_by_years.agg({"WeightedPositiveScore": "sum", "TotalPositiveResponses": "sum"})

        # Step 5: Calculate the overall average rating
        overall_positive_average_rating = (overall_totals_positive["WeightedPositiveScore"] / overall_totals_positive["TotalPositiveResponses"])
        overall_positive_rating_percent = overall_positive_average_rating/5*100

        st.markdown(f"<div style='text-align: center; font-size:45px; color:red'>{overall_positive_rating_percent:.1f}%</div>", unsafe_allow_html=True) 
        st.markdown(f"<div style='text-align: center; font-size:16px;'>Average Favorability</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size:14px;'> <strong>{overall_positive_average_rating:.2f}</strong> out of 5 </div>", unsafe_allow_html=True)

   
    col2_1, col2_2 = st.columns((1,3))

    with col2_1:
        select_metric = st.selectbox("",options=["SFT Score", "Favorability"])
        st.markdown("**SFT Score:** The average score attained by the faculty based on all student feedback ratings collected.")
        st.markdown("**Favorability:** The average score attained by the faculty based on all _positive_ student feedback ratings collected.")
        st.markdown("\* The overall average favorability data is unavailable.")
    with col2_2:
        if select_metric == "SFT Score":
            # Step 1: Sum the columns for feedback (all responses)
            filtered_df["TotalResponses"] = (
                filtered_df["Strongly Agree"]
                + filtered_df["Agree"]
                + filtered_df["Neutral"]
                + filtered_df["Disagree"]
                + filtered_df["Strongly Disagree"]
            )

            # Step 2: Calculate the weighted score
            filtered_df["WeightedScore"] = filtered_df["Score"] * filtered_df["TotalResponses"]

        elif select_metric == "Favorability":
            # Step 1: Sum the columns for feedback (all responses)
            filtered_df["TotalResponses"] = (
                filtered_df["Strongly Agree"]
                + filtered_df["Agree"]
            )

            # Step 2: Calculate the weighted score
            filtered_df["WeightedScore"] = filtered_df["Favor"] * filtered_df["TotalResponses"]

        # Step 3: Group by `AcadYear` and `Semester` and calculate totals
        semester_totals = (
            filtered_df.groupby(["AcadYear", "Semester"], as_index=False)
            .agg({"WeightedScore": "sum", "TotalResponses": "sum"})
        )

        # Step 4: Calculate the average rating for each year-semester combination
        semester_totals["AverageRating"] = semester_totals["WeightedScore"] / semester_totals["TotalResponses"]

        # Step 5: Combine `AcadYear` and `Semester` for labeling on the x-axis
        semester_totals["YearSemester"] = semester_totals["AcadYear"].astype(str).str.replace("\.0", "", regex=True) + " " + semester_totals["Semester"].astype(str)
        
        # Step 6: Define color categories for years
        def assign_color(year):
            if year == 2016:
                return "Undergraduate TA"
            elif 2017 <= year <= 2018:
                return "Project Officer (TEL)"
            elif 2023 <= year <= 2024:
                return "Part-time Instructor"
            else:
                return "Others"  # Default for other years
        
        semester_totals["ColorCategory"] = semester_totals["AcadYear"].apply(assign_color)

        # Step 7: Map colors to categories
        color_map = {
            "Undergraduate TA": "lightblue",
            "Project Officer (TEL)": "lightgreen",
            "Part-time Instructor": "lightcoral",
            "Others": "gray",
        }

        # Step 8: Plot the data using Plotly bar chart
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
        st.plotly_chart(fig)
    
with tabs[1]:
    st.header("Detailed Reports")

    subtab_names = ["**Criteria Report**", "**Course Report**"]
    subtabs = st.tabs(subtab_names)

    with subtabs[0]:
        st.subheader("Criteria-level Analysis")

        # Multi-select for filtering
        unique_criteria = SFT_criteria["Criteria Shortname"].dropna().unique()
        selected_criteria = st.multiselect("", options=unique_criteria, default=unique_criteria)

        min_year, max_year = year_range

        # Filter to only include min and max years
        filtered_df["AcadYear"] = filtered_df["AcadYear"].astype(str).str.replace("\.0", "", regex=True)
        filtered_df = filtered_df[
            (filtered_df["AcadYear"].str.replace("\.0", "", regex=True).isin([str(min_year), str(max_year)])) &
            (filtered_df["Criteria"].isin(selected_criteria))
            ]

        # Calculate average scores for each year and domain
        average_scores = filtered_df.groupby(["AcadYear", "Criteria"])["Score"].mean().unstack()
        favor_scores = filtered_df.groupby(["AcadYear", "Criteria"])["Favor"].mean().unstack()

        # Check if data is available for the selected min_year and max_year
        if str(min_year) not in average_scores.index or str(max_year) not in average_scores.index:
            st.warning("Data not available for the selected start/end year.")
        else:
            # Prepare data for plotting
            fig1 = go.Figure()
            # Define colors for each domain
            num_criteria = len(unique_criteria)
            generated_colors = pc.qualitative.Dark24[:num_criteria]
            dynamic_color_palette = {criteria: generated_colors[i] for i, criteria in enumerate(unique_criteria)}
    
            # Add a line for each domain
            for i, domain in enumerate(average_scores.columns):
                # Get values for min and max years
                min_value = average_scores.loc[str(min_year), domain] if str(min_year) in average_scores.index else None
                max_value = average_scores.loc[str(max_year), domain] if str(max_year) in average_scores.index else None
                
                # Determine line style (solid for increase, dashed for decrease)
                line_style = "solid" if pd.notna(min_value) and pd.notna(max_value) and max_value >= min_value else "dash"
                
                # Use the custom color palette for the domain
                color = dynamic_color_palette.get(domain, "#333333")

                fig1.add_trace(
                    go.Scatter(
                        x=[0,1],
                        y=average_scores.loc[[str(min_year), str(max_year)], domain].values,
                        mode="lines+markers",
                        name=domain,
                        marker=dict(size=10),
                        line=dict(width=2, dash=line_style, color=color),
                    )
                )

            # Customize the layout
            fig1.update_layout(
                xaxis=dict(
                    tickvals=[0,1],
                    ticktext=[str(min_year), str(max_year)],
                    title=f"Criteria Average Changes between {min_year} and {max_year}"
                ),
                yaxis=dict(title="Average Score"),
                legend_title="Domains",
            )

            # Add a line for each domain
            for i, domain in enumerate(average_scores.columns):
                # Get values for min and max years
                min_value = average_scores.loc[str(min_year), domain] if str(min_year) in average_scores.index else None
                max_value = average_scores.loc[str(max_year), domain] if str(max_year) in average_scores.index else None
                
                # Skip if both values are NaN
                if pd.isna(min_value) and pd.isna(max_value):
                    continue
                
                # Use the custom color palette for the domain
                color = dynamic_color_palette.get(domain, "#333333")

                # Prepare x and y data, skipping NaNs
                x_data = []
                y_data = []
                if not pd.isna(min_value):
                    x_data.append(0)  # Min year position
                    y_data.append(min_value)
                if not pd.isna(max_value):
                    x_data.append(1)  # Max year position
                    y_data.append(max_value)

                # Add annotations only for non-NaN values
                if not pd.isna(min_value):
                    # Min Year Annotation (Right-Aligned)
                    fig1.add_annotation(
                        x=0-0.02,
                        y=min_value,
                        text=f"{domain} ({min_value:.2f})",
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xanchor="right",
                    )
                if not pd.isna(max_value):
                    # Max Year Annotation (Left-Aligned)
                    fig1.add_annotation(
                        x=1+0.01,
                        y=max_value,
                        text=f"{domain} ({max_value:.2f})",
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xanchor="left",
                    )

            # Add a vertical line for the min_year
            fig1.add_shape(
                type="line",
                x0=0,  # Normalized x-coordinate for min_year
                x1=0,  # Same x-coordinate for a vertical line
                y0=0,  # Start at the bottom of the plot
                y1=1,  # End at the top of the plot
                xref="x",  # Reference the x-axis
                yref="paper",  # Reference the full height of the plot
                line=dict(
                    color="gray",  # Line color
                    width=1,        # Line width
                    dash="dot",     # Dotted line style
                ),
                layer="below",  # Draw the line behind all text and plots
            )

            # Add a vertical line for the max_year
            fig1.add_shape(
                type="line",
                x0=1,  # Normalized x-coordinate for max_year
                x1=1,  # Same x-coordinate for a vertical line
                y0=0,  # Start at the bottom of the plot
                y1=1,  # End at the top of the plot
                xref="x",  # Reference the x-axis
                yref="paper",  # Reference the full height of the plot
                line=dict(
                    color="gray",  # Line color
                    width=1,        # Line width
                    dash="dot",     # Dotted line style
                ),
                layer="below",  # Draw the line behind all text and plots
            )
            # Display the plot in Streamlit
            st.plotly_chart(fig1, use_container_width=True)

            # Prepare data for plotting
            fig2 = go.Figure()
            # Define colors for each domain
            num_criteria = len(unique_criteria)
            generated_colors = pc.qualitative.Dark24[:num_criteria]
            dynamic_color_palette = {criteria: generated_colors[i] for i, criteria in enumerate(unique_criteria)}
    
            # Add a line for each domain
            for i, domain in enumerate(favor_scores.columns):
                # Get values for min and max years
                min_value = favor_scores.loc[str(min_year), domain] if str(min_year) in favor_scores.index else None
                max_value = favor_scores.loc[str(max_year), domain] if str(max_year) in favor_scores.index else None
                
                # Determine line style (solid for increase, dashed for decrease)
                line_style = "solid" if pd.notna(min_value) and pd.notna(max_value) and max_value >= min_value else "dash"
                
                # Use the custom color palette for the domain
                color = dynamic_color_palette.get(domain, "#333333")

                fig2.add_trace(
                    go.Scatter(
                        x=[0,1],
                        y=favor_scores.loc[[str(min_year), str(max_year)], domain].values,
                        mode="lines+markers",
                        name=domain,
                        marker=dict(size=10),
                        line=dict(width=2, dash=line_style, color=color),
                    )
                )

            # Customize the layout
            fig2.update_layout(
                xaxis=dict(
                    tickvals=[0,1],
                    ticktext=[str(min_year), str(max_year)],
                    title=f"Criteria Favorability Changes between {min_year} and {max_year}"
                ),
                yaxis=dict(title="Favorability Score"),
                legend_title="Domains",
            )

            # Add a line for each domain
            for i, domain in enumerate(favor_scores.columns):
                # Get values for min and max years
                min_value = favor_scores.loc[str(min_year), domain] if str(min_year) in favor_scores.index else None
                max_value = favor_scores.loc[str(max_year), domain] if str(max_year) in favor_scores.index else None
                
                # Skip if both values are NaN
                if pd.isna(min_value) and pd.isna(max_value):
                    continue
                
                # Use the custom color palette for the domain
                color = dynamic_color_palette.get(domain, "#333333")

                # Prepare x and y data, skipping NaNs
                x_data = []
                y_data = []
                if not pd.isna(min_value):
                    x_data.append(0)  # Min year position
                    y_data.append(min_value)
                if not pd.isna(max_value):
                    x_data.append(1)  # Max year position
                    y_data.append(max_value)

                # Add annotations only for non-NaN values
                if not pd.isna(min_value):
                    # Min Year Annotation (Right-Aligned)
                    fig2.add_annotation(
                        x=0-0.02,
                        y=min_value,
                        text=f"{domain} ({min_value:.2f})",
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xanchor="right",
                    )
                if not pd.isna(max_value):
                    # Max Year Annotation (Left-Aligned)
                    fig2.add_annotation(
                        x=1+0.01,
                        y=max_value,
                        text=f"{domain} ({max_value:.2f})",
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xanchor="left",
                    )

            # Add a vertical line for the min_year
            fig2.add_shape(
                type="line",
                x0=0,  # Normalized x-coordinate for min_year
                x1=0,  # Same x-coordinate for a vertical line
                y0=0,  # Start at the bottom of the plot
                y1=1,  # End at the top of the plot
                xref="x",  # Reference the x-axis
                yref="paper",  # Reference the full height of the plot
                line=dict(
                    color="gray",  # Line color
                    width=1,        # Line width
                    dash="dot",     # Dotted line style
                ),
                layer="below",  # Draw the line behind all text and plots
            )

            # Add a vertical line for the max_year
            fig2.add_shape(
                type="line",
                x0=1,  # Normalized x-coordinate for max_year
                x1=1,  # Same x-coordinate for a vertical line
                y0=0,  # Start at the bottom of the plot
                y1=1,  # End at the top of the plot
                xref="x",  # Reference the x-axis
                yref="paper",  # Reference the full height of the plot
                line=dict(
                    color="gray",  # Line color
                    width=1,        # Line width
                    dash="dot",     # Dotted line style
                ),
                layer="below",  # Draw the line behind all text and plots
            )
            # Display the plot in Streamlit
            st.plotly_chart(fig2, use_container_width=True)

    with subtabs[1]:
        # Sorting and filtering courses
        SFT_courses_sorted = SFT_courses.sort_values(by="CourseCode", ascending=True)
        filtered_course = SFT_courses_sorted[(SFT_courses_sorted["AcadYear"].isin(years)) & 
                                            (SFT_courses_sorted["Stage"].isin(course_level)) & 
                                            (SFT_courses_sorted["Discipline"].isin(subject_domain))]
        selected_course = st.selectbox("", (filtered_course["CourseCode"] + " - " + filtered_course["CourseName"]).unique())

        # Combine TutorialGrp and AcadYear into a unique identifier
        df["TutorialGrp_Year"] = df["TutorialGrp"] + " (" + df["AcadYear"].astype(str).str.replace("\.0", "", regex=True) + ")"

        # Filter df based on the selected course
        filtered_df = df[(df["CourseCode"] + " - " + df["CourseName"] == selected_course)]

        # Grouping data by Criteria and TutorialGrp_Year
        aggregated_scores = filtered_df.groupby(['Criteria', 'TutorialGrp_Year'])['Score'].mean().unstack()
        aggr_scores_mean = filtered_df.groupby(['Criteria'])['Score'].mean().sort_values(ascending=False).index
        aggregated_scores = aggregated_scores.loc[aggr_scores_mean]
        aggregated_favor = filtered_df.groupby(['Criteria', 'TutorialGrp_Year'])['Favor'].mean().unstack()
        aggr_favor_mean = filtered_df.groupby(['Criteria'])['Favor'].mean().sort_values(ascending=False).index
        aggregated_favor = aggregated_favor.loc[aggr_favor_mean]
        
        # Sort tutorials within each domain so earlier years comes first
        tutorial_years = {grp: int(grp.split('(')[-1].strip('.0)')) for grp in aggregated_scores.columns}  # Extract year
        aggregated_scores = aggregated_scores[sorted(aggregated_scores.columns, key=lambda x: (tutorial_years[x], x))]
        aggregated_favor = aggregated_favor[aggregated_scores.columns]

        # Prepare indices and colors
        indices = np.arange(len(aggregated_scores))
        unique_tutorials = aggregated_scores.columns
        colors = px.colors.qualitative.Pastel

        # Create Plotly figure for Scores
        fig_scores = go.Figure()

        # Plot Scores
        for i, tutorial in enumerate(aggregated_scores.columns):
            fig_scores.add_trace(go.Bar(
                x=indices + i * 0.1,  # Bar positions
                y=aggregated_scores[tutorial],
                name=tutorial,
                marker=dict(color=colors[i % len(colors)]),
                width=0.1
            ))

        # Annotate mean scores
        for idx, (mean_score, stdev_score) in enumerate(zip(aggregated_scores.mean(axis=1), filtered_df.groupby(['Criteria'])['ScoreStdev'].mean())):
            fig_scores.add_annotation(
                x=indices[idx],
                y=5.1,
                text=f"{mean_score:.2f} ± {stdev_score:.2f}", xanchor="left",
                showarrow=False,
                font=dict(size=12, color="black")
            )

        # Update layout for Scores
        fig_scores.update_layout(
            barmode="group",
            xaxis=dict(
                title="Criteria",
                tickvals=indices,
                ticktext=aggregated_scores.index,
                tickangle=45
            ),
            yaxis=dict(title="Scores", range=[3.9, 5.1], dtick=0.1, minor=dict(ticks='outside', tick0=4, dtick=0.05)),
            legend_title="Tutorial Group",
            template="plotly_white"
        )

        # Create Plotly figure for Favorability
        fig_favor = go.Figure()

        # Plot Favorability
        for i, tutorial in enumerate(aggregated_favor.columns):
            fig_favor.add_trace(go.Bar(
                x=indices + i * 0.1,  # Bar positions
                y=aggregated_favor[tutorial],
                name=tutorial,
                marker=dict(color=colors[i % len(colors)]),
                width=0.1
            ))

        # Annotate mean favorability
        for idx, (mean_favor, stdev_favor) in enumerate(zip(aggregated_favor.mean(axis=1), filtered_df.groupby(['Criteria'])['FavStdev'].mean())):
            fig_favor.add_annotation(
                x=indices[idx],
                y=5.1,
                text=f"{mean_favor:.2f} ± {stdev_favor:.2f}", xanchor="left",
                showarrow=False,
                font=dict(size=12, color="black")
            )

        # Update layout for Favorability
        fig_favor.update_layout(
            barmode="group",
            xaxis=dict(
                title="Criteria",
                tickvals=indices,
                ticktext=aggregated_favor.index,
                tickangle=45
            ),
            yaxis=dict(title="Favorability", range=[3.9, 5.1], dtick=0.1, minor=dict(ticks='outside', tick0=4, dtick=0.05)),
            legend_title="Tutorial Group",
            template="plotly_white"
        )

        # Display the figures in Streamlit
        st.plotly_chart(fig_scores)
        st.plotly_chart(fig_favor)
