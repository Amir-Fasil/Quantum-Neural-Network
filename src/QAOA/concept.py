class Concept:
    
    def __init__(self, extent: set, intent: set):

        self.extent = extent
        self.intent = intent    

    def get_extent(self):

        """Return the extent of the concept."""
        return self.extent
    
    def get_intent(self):

        """Return the intent of the concept."""
        return self.intent
    
    def __repr__(self):

        return f"CONCEPT(Extent:{self.extent}, Intent:{self.intent})"
    
    def get_Concept(self):
        
        """Return the concept as a tuple (extent, intent)."""
        return (self.get_extent(), self.get_intent())