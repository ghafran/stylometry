"""Source bibliography and undisputed Federalist benchmark labels."""
FED_DISPUTED = {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63}
FED_JOINT = {18, 19, 20}
FED_JAY = {2, 3, 4, 5, 64}
FED_MADISON = {10, 14, *range(37, 49)}

# (gutenberg id, author, short code, title, collection, genre)
CATALOGUE: list[tuple[int, str, str, str, str, str]] = [
    # --- Novels: one genre, several works per author -------------------------------------------------
    (1342, "Jane Austen", "AUSTEN-PP", "Pride and Prejudice", "Novels", "fiction"),
    (161, "Jane Austen", "AUSTEN-SS", "Sense and Sensibility", "Novels", "fiction"),
    (158, "Jane Austen", "AUSTEN-EM", "Emma", "Novels", "fiction"),
    (1400, "Charles Dickens", "DICKENS-GE", "Great Expectations", "Novels", "fiction"),
    (98, "Charles Dickens", "DICKENS-TC", "A Tale of Two Cities", "Novels", "fiction"),
    (730, "Charles Dickens", "DICKENS-OT", "Oliver Twist", "Novels", "fiction"),
    (1260, "Charlotte Bronte", "BRONTE-JE", "Jane Eyre", "Novels", "fiction"),
    (9182, "Charlotte Bronte", "BRONTE-VI", "Villette", "Novels", "fiction"),
    (30486, "Charlotte Bronte", "BRONTE-SH", "Shirley", "Novels", "fiction"),
    (145, "George Eliot", "ELIOT-MM", "Middlemarch", "Novels", "fiction"),
    (550, "George Eliot", "ELIOT-SM", "Silas Marner", "Novels", "fiction"),
    (6688, "George Eliot", "ELIOT-MF", "The Mill on the Floss", "Novels", "fiction"),
    (110, "Thomas Hardy", "HARDY-TD", "Tess of the d'Urbervilles", "Novels", "fiction"),
    (27, "Thomas Hardy", "HARDY-FM", "Far from the Madding Crowd", "Novels", "fiction"),
    (153, "Thomas Hardy", "HARDY-JO", "Jude the Obscure", "Novels", "fiction"),
    # --- Cross-genre: the same hand in fiction and in prose ------------------------------------------
    (74, "Mark Twain", "TWAIN-TS", "The Adventures of Tom Sawyer", "Cross-genre", "fiction"),
    (76, "Mark Twain", "TWAIN-HF", "Adventures of Huckleberry Finn", "Cross-genre", "fiction"),
    (245, "Mark Twain", "TWAIN-LM", "Life on the Mississippi", "Cross-genre", "non-fiction"),
    (3176, "Mark Twain", "TWAIN-IA", "The Innocents Abroad", "Cross-genre", "non-fiction"),
    (1695, "G. K. Chesterton", "CHESTERTON-MT", "The Man Who Was Thursday", "Cross-genre", "fiction"),
    (204, "G. K. Chesterton", "CHESTERTON-FB", "The Innocence of Father Brown", "Cross-genre", "fiction"),
    (470, "G. K. Chesterton", "CHESTERTON-HE", "Heretics", "Cross-genre", "non-fiction"),
    (16769, "G. K. Chesterton", "CHESTERTON-OR", "Orthodoxy", "Cross-genre", "non-fiction"),
    (120, "Robert Louis Stevenson", "STEVENSON-TI", "Treasure Island", "Cross-genre", "fiction"),
    (43, "Robert Louis Stevenson", "STEVENSON-JH", "Dr Jekyll and Mr Hyde", "Cross-genre", "fiction"),
    (535, "Robert Louis Stevenson", "STEVENSON-TD", "Travels with a Donkey", "Cross-genre", "non-fiction"),
    (386, "Robert Louis Stevenson", "STEVENSON-VP", "Virginibus Puerisque", "Cross-genre", "non-fiction"),
    (36, "H. G. Wells", "WELLS-WW", "The War of the Worlds", "Cross-genre", "fiction"),
    (35, "H. G. Wells", "WELLS-TM", "The Time Machine", "Cross-genre", "fiction"),
    (19229, "H. G. Wells", "WELLS-AN", "Anticipations", "Cross-genre", "non-fiction"),
    (7058, "H. G. Wells", "WELLS-MM", "Mankind in the Making", "Cross-genre", "non-fiction"),
    (1661, "Arthur Conan Doyle", "DOYLE-SH", "The Adventures of Sherlock Holmes", "Cross-genre", "fiction"),
    (2852, "Arthur Conan Doyle", "DOYLE-HB", "The Hound of the Baskervilles", "Cross-genre", "fiction"),
    (3069, "Arthur Conan Doyle", "DOYLE-BW", "The Great Boer War", "Cross-genre", "non-fiction"),
    (5317, "Arthur Conan Doyle", "DOYLE-MD", "Through the Magic Door", "Cross-genre", "non-fiction"),
]

FEDERALIST_ID = 1404
