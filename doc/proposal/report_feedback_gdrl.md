# Proposal Report Grading Feedback

- Partner: Gerontology Diabetes Research Lab (GDRL)
- Title: Diagnosing Lipohypertrophy
- Team: Ela Bandari, Lara Habashy, Javairia Raza, and Peter Yang

## Reasoning (6.8/7)

- Overall the content of the report is excellent, it was a pleasure to read and a solid summary of what you've done and what you're planning for this project
- The only thing I'm missing in this report is a short summary of any similar work (if any) that has been done before. For example, have other studies/competitions/projects looked at identifying elements in ultrasound/x-ray images, are there any pre-existing networks that could be used that have been specifically trained on this style of image?

## Viz (0.9/1)

- Good quality viz, with caption, title, colorbar, etc.
- Only thing missing on these ones is a scale which could be useful
- Perhaps make the titles a little larger on the matplotlib plots
- The timeline figure should still have a caption, you can add one easily using this syntax: https://jupyterbook.org/content/figures.html#figures
- The only problem is that you're creating the figure caption manually with matplotlib, and the numbering will be different to what Jupyter Book generates (using the method in the link above). I see two solutions which we can discuss as we get closer to the final report:
  - 1. Have a script generate the plots of the ultrasound images and save them as a .png which is then incorporated into the book using the {figure} directive shown in the link above
  - 2. We could potentially create a "dummy figure" below the ultrasound images which just has the caption metadata but no figure

## Writing (0.95/1)

- Excellent writing, the text is accessible to our non-technical partner, but still descriptive in the data science techniques and approach
- Incorporated feedback well
- Report is clearly and logically organised
- Only comment is to make the in-text citations more readable. For example, instead of `[Mad]` it would be nice to see `[Madden, 2021]` (you should be able to change the format using the instructions here: https://jupyterbook.org/content/citations.html#change-the-in-line-citation-style)

## Mechanics (0.95/1)

- Excellent use of Jupyter Book!
- Cool to see the code in there (hidden). For full reproducibility we should also generate the table of postivie/negative counts programmtically. We could do that easily by populating a dataframe for example and printing it. We want to populate the dataframe programatically too. There are a few tools to count how many files are in a directory - `os.listdir()` is an easy one. In this way, if we add more images to a folder, the report will automatically update when re-built and you won't have to manually modify that table.
- For the final report, we may prefer to build a single page rather than a book of multiple pages. You can do that following these instructions: https://jupyterbook.org/basics/page.html
- We'll also want to build a PDF (but that's something to think about later on): https://jupyterbook.org/advanced/pdf.html

## Final Grade (9.9/10)

- Fantastic work team. You continue to be one of the top performing capstone teams and I, and Ken, are lucky to have you. It's only been 2 weeks and you've done amazing work. I can't wait to see what amazing things you come up with by the end of the project!