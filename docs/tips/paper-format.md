# 论文格式

## ACM Primary Article Template (`acmart`)

- [ACM Primary Article Template](https://www.acm.org/publications/proceedings-template)

### Template Styles (`\documentclass[STYLE]{acmart}`)

- Journal
  - `acmsmall`: The default journal template style.
  - `acmlarge`: Used by JOCCH and TAP.
  - `acmtog`: Used by TOG.
- Conference
  - `sigconf`: The default proceedings template style. 
  - `sigchi`: Used for SIGCHI conference articles.
  - `sigplan`: Used for SIGPLAN conference articles.

### Template Parameters

- `anonymous`,`review`: Suitable for a “double-anonymous” conference submission. Anonymizes the work and includes line numbers. Use with the command to print the submission’s unique ID on each page of the work.
- `authorversion`: Produces a version of the work suitable for posting by the author. 
- `screen`: Produces colored hyperlinks.

### MODIFICATIONS

Modifying the template — including but not limited to: adjusting margins, typeface sizes, line spacing, paragraph and list definitions, and the use of the `\vspace` command to manually adjust the vertical spacing between elements of your work — is not allowed.

Your document will be returned to you for revision if modifications are discovered.

### TYPEFACES

The “acmart” document class requires the use of the “`Libertine`” typeface family. Your TEX installation should include this set of packages. Please do not substitute other typefaces. The “`lmodern`” and “`ltimes`” packages <font color=red>should not be used</font>, as they will override the built-in typeface families.

### TITLE INFORMATION

The title of your work should use capital letters appropriately - https://capitalizemytitle.com/ has useful rules for capitalization. Use the title command to define the title of your work. If your work has a subtitle, define it with the subtitle command. Do not insert line breaks in your title.

If your title is lengthy, you <font color=red>must define</font> a short version to be used in the page headers, to prevent overlapping text. The title command has a “short title” parameter:

```
\title[short title]{full title}
```

### AUTHORS AND AFFILIATIONS

Each author must be defined separately for accurate metadata identification. As an exception, multiple authors may share one affiliation. Authors’ names should not be abbreviated; use full first names wherever possible. Include authors’ e-mail addresses whenever possible.

Grouping authors’ names or e-mail addresses, or providing an “e-mail alias,” as shown below, is not acceptable:

```
\author{Brooke Aster, David Mehldau}
\email{dave,judy,steve@university.edu}
\email{firstname.lastname@phillips.org}
```

The authornote and authornotemark commands allow a note to apply to multiple authors — for example, if the first two authors of an article contributed equally to the work.

If your author list is lengthy, you must define a shortened version of the list of authors to be used in the page headers, to prevent overlapping text. The following command should be placed just after the last `\author{}` definition:

```
\renewcommand{\shortauthors}{McCartney, et al.}
```

Omitting this command will force the use of a concatenated list of all of the authors’ names, which may result in overlapping text in the page headers.

The article template’s documentation, available at https://www.acm.org/publications/proceedings-template, has a complete explanation of these commands and tips for their effective use.

Note that authors’ addresses are mandatory for journal articles.

### RIGHTS INFORMATION

Authors of any work published by ACM will need to complete a rights form. Depending on the kind of work, and the rights management choice made by the author, this may be copyright transfer, permission, license, or an OA (open access) agreement.

Regardless of the rights management choice, the author will receive a copy of the completed rights form once it has been submitted. This form contains LATEX commands that must be copied into the source document. When the document source is compiled, these commands and their parameters add formatted text to several areas of the final document:

- the “ACM Reference Format” text on the first page.
- the “rights management” text on the first page.
- the conference information in the page header(s).

Rights information is unique to the work; if you are preparing several works for an event, make sure to use the correct set of commands with each of the works.

The ACM Reference Format text is required for all articles over one page in length, and is optional for one-page articles (abstracts).

### CCS CONCEPTS AND USER-DEFINED KEYWORDS

Two elements of the “acmart” document class provide powerful taxonomic tools for you to help readers find your work in an online search.

The ACM Computing Classification System — https://www.acm.org/publications/class-2012 — is a set of classifiers and concepts that describe the computing discipline. Authors can select entries from this classification system, via https://dl.acm.org/ccs/ccs.cfm, and generate the commands to be included in the LATEX source.

User-defined keywords are a comma-separated list of words and phrases of the authors’ choos- ing, providing a more flexible way of describing the research being presented.

CCS concepts and user-defined keywords are required for for all articles over two pages in length, and are optional for one- and two-page articles (or abstracts).

### SECTIONING COMMANDS

Your work should use standard LATEX sectioning commands: `section`, `subsection`, `subsubsection`, and `paragraph`. They should be numbered; do not remove the numbering from the commands.

Simulating a sectioning command by setting the first word or words of a paragraph in boldface or italicized text is <font color=red>not allowed</font>.

### TABLES

The “`acmart`” document class includes the “booktabs” package — https://ctan.org/pkg/booktabs — for preparing high-quality tables.

Table captions are placed above the table.

Because tables cannot be split across pages, the best placement for them is **typically the top of the page** nearest their initial cite. To ensure this proper “floating” placement of tables, use the environment `table` to enclose the table’s contents and the table caption. The contents of the table itself must go in the `tabular` environment, to be aligned properly in rows and columns, with the desired horizontal and vertical rules. Again, detailed instructions on tabular material are found in the LATEX User’s Guide.

To set a wider table, which takes up the whole width of the page’s live area, use the environment `table*` to enclose the table’s contents and the table caption. As with a single-column table, this wide table will “float” to a location deemed more desirable. 

Always use `midrule` to separate table header rows from data rows, and use it only for this purpose. This enables assistive technologies to recognise table headers and support their users in navigating tables more easily.

### FIGURES

The “figure” environment should be used for figures. One or more images can be placed within a figure. If your figure contains third-party material, you must clearly identify it as such: 

```
Fig. 1. 1907 Franklin Model D roadster. Photograph by Harris & Ewing, Inc. [Public domain], via Wikimedia Commons. (https://goo.gl/VLCRBB).
```

Your figures should contain a caption which describes the figure to the reader.

Figure captions are placed below the figure.

Every figure should also have a figure description unless it is purely decorative. These descriptions convey what’s in the image to someone who cannot see it. They are also used by search engine crawlers for indexing images, and when images cannot be loaded.

A figure description must be unformatted plain text less than 2000 characters long (including spaces). Figure descriptions should not repeat the figure caption — their purpose is to capture important information that is not already provided in the caption or the main text of the paper. For figures that convey important and complex new information, a short text description may not be adequate. More complex alternative descriptions can be placed in an appendix and referenced in a short figure description. For example, provide a data table capturing the information in a bar chart, or a structured list representing a graph. For additional information regarding how best to write figure descriptions and why doing this is so important, please see https://www.acm.org/publications/taps/describing-figures/.

#### The “Teaser Figure”

A “teaser figure” is an image, or set of images in one figure, that are placed after all author and affiliation information, and before the body of the article, spanning the page. If you wish to have such a figure in your article, place the command immediately before the `\maketitle` command:

```
\begin{teaserfigure}
    \includegraphics[width=\textwidth]{sampleteaser}
    \caption{figure caption}
    \Description{figure description}
\end{teaserfigure}
```

### CITATIONS AND BIBLIOGRAPHIES

The use of BibTEX for the preparation and formatting of one’s references is strongly recommended. Authors’ names should be complete — use full first names (“Donald E. Knuth”) not initials (“D. E. Knuth”) — and the salient identifying features of a reference should be included: title, year, volume, number, pages, article DOI, etc.

The bibliography is included in your source document with these two commands, placed just before the `\end{document}` command:	

```
\bibliographystyle{ACM-Reference-Format}
\bibliography{bibfile}
```

where “`bibfile`” is the name, without the “`.bib`” suffix, of the BibTEX file.

Citations and references are numbered by default. A small number of ACM publications have citations and references formatted in the “author year” style; for these exceptions, please include this command in the preamble (before the command “`\begin{document}`”) of your LATEX source:

```
\citestyle{acmauthoryear}
```

### ACKNOWLEDGMENTS

Identification of funding sources and other support, and thanks to individuals and groups that assisted in the research and the preparation of the work should be included in an acknowledgment section, which is placed just before the reference section in your document.

This section has a special environment:

```
\begin{acks}
...
\end{acks}
```

so that the information contained therein can be more easily collected during the article metadata extraction phase, and to ensure consistency in the spelling of the section heading.

**Authors should not prepare this section as a numbered or unnumbered `\section`; please use the “`acks`” environment.**

### APPENDICES

If your work needs an appendix, add it before the “`\end{document}`” command at the conclusion of your source document.

Start the appendix with the “`\appendix`” command, and note that in the appendix, sections are lettered, not numbered.

## USENIX

- [Templates for Conference Papers](https://www.usenix.org/conferences/author-resources/paper-templates)
