import sys
from crossword import *
from PIL import Image, ImageDraw, ImageFont


class CrosswordSolver:
    def __init__(self, puzzle):
        """Initialize crossword solver with given puzzle structure and word list."""
        self.puzzle = puzzle
        self.domains = {var: puzzle.words.copy() for var in puzzle.variables}

    def generate_grid(self, assignment):
        """Construct a 2D representation of the crossword assignment."""
        grid = [[None for _ in range(self.puzzle.width)] for _ in range(self.puzzle.height)]
        for var, word in assignment.items():
            for index, letter in enumerate(word):
                row = var.i + (index if var.direction == Variable.DOWN else 0)
                col = var.j + (index if var.direction == Variable.ACROSS else 0)
                grid[row][col] = letter
        return grid

    def display(self, assignment):
        """Print the crossword assignment in a readable format."""
        grid = self.generate_grid(assignment)
        for i in range(self.puzzle.height):
            for j in range(self.puzzle.width):
                print(grid[i][j] or ("█" if not self.puzzle.structure[i][j] else " "), end="")
            print()

    def save_as_image(self, assignment, filename):
        """Save the crossword as an image file."""
        cell_size = 100
        border_size = 2
        grid = self.generate_grid(assignment)
        img = Image.new("RGBA", (self.puzzle.width * cell_size, self.puzzle.height * cell_size), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)

        for i in range(self.puzzle.height):
            for j in range(self.puzzle.width):
                if self.puzzle.structure[i][j]:
                    rect = [(j * cell_size + border_size, i * cell_size + border_size),
                            ((j + 1) * cell_size - border_size, (i + 1) * cell_size - border_size)]
                    draw.rectangle(rect, fill="white")
                    if grid[i][j]:
                        draw.text((rect[0][0] + 30, rect[0][1] + 10), grid[i][j], fill="black", font=font)

        img.save(filename)

    def enforce_constraints(self):
        """Enforce node and arc consistency and solve the crossword."""
        self.apply_node_consistency()
        self.apply_arc_consistency()
        return self.solve(dict())

    def apply_node_consistency(self):
        """Remove words that do not match the length of the variable."""
        for var in list(self.domains):
            self.domains[var] = {word for word in self.domains[var] if len(word) == var.length}

    def apply_arc_consistency(self):
        """Ensure arc consistency across the puzzle variables."""
        queue = [(x, y) for x in self.domains for y in self.puzzle.neighbors(x)]
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                queue.extend((z, x) for z in self.puzzle.neighbors(x) if z != y)
        return True

    def revise(self, x, y):
        """Remove values from x's domain that are inconsistent with y."""
        if (overlap := self.puzzle.overlaps.get((x, y))) is None:
            return False
        i, j = overlap
        to_remove = {word for word in self.domains[x] if all(word[i] != other[j] for other in self.domains[y])}
        self.domains[x] -= to_remove
        return bool(to_remove)

    def solve(self, assignment):
        """Perform backtracking search to find a valid crossword assignment."""
        if self.is_complete(assignment):
            return assignment
        var = self.select_variable(assignment)
        for value in sorted(self.domains[var], key=lambda val: self.get_constraints(var, val)):
            assignment[var] = value
            if self.is_consistent(assignment):
                result = self.solve(assignment)
                if result:
                    return result
            assignment.pop(var)
        return None

    def is_complete(self, assignment):
        """Check if the assignment is complete."""
        return len(assignment) == len(self.puzzle.variables)

    def is_consistent(self, assignment):
        """Check if the assignment is valid based on crossword rules."""
        words_used = set()
        for var, word in assignment.items():
            if word in words_used or len(word) != var.length:
                return False
            words_used.add(word)
            for neighbor in self.puzzle.neighbors(var):
                if neighbor in assignment:
                    overlap = self.puzzle.overlaps.get((var, neighbor))
                    if overlap and assignment[var][overlap[0]] != assignment[neighbor][overlap[1]]:
                        return False
        return True

    def select_variable(self, assignment):
        """Select the next variable to assign, prioritizing constraints."""
        unassigned = [var for var in self.puzzle.variables if var not in assignment]
        return min(unassigned, key=lambda v: (len(self.domains[v]), -len(self.puzzle.neighbors(v))))

    def get_constraints(self, var, word):
        """Count how many values this word eliminates in neighboring variables."""
        return sum(1 for neighbor in self.puzzle.neighbors(var) if word in self.domains[neighbor])


def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    puzzle = Crossword(sys.argv[1], sys.argv[2])
    solver = CrosswordSolver(puzzle)
    assignment = solver.enforce_constraints()

    if assignment is None:
        print("No solution.")
    else:
        solver.display(assignment)
        if len(sys.argv) == 4:
            solver.save_as_image(assignment, sys.argv[3])


if __name__ == "__main__":
    main()
