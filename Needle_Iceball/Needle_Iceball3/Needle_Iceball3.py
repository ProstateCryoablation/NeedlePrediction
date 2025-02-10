import logging
import os
from typing import Annotated, Optional

import vtk, qt, ctk, slicer

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import numpy as np
import torch
from scipy.stats import chi2
import pandas as pd
from PyQt5.QtCore import QTimer, QCoreApplication
import time
import SimpleITK as sitk
import sitkUtils
from vtk.util import numpy_support
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from collections import defaultdict
#
# Iceball_needle3
#


class Needle_Iceball3(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Needle_Iceball3")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Eva Beek (Surgical Planning Laboratory)"]
        self.parent.helpText = _("""This module places targets within a volume and ensures they do not overlap with other targets.""")
        self.parent.acknowledgementText = _("""Original development by...""")

    def initializeParameterNode(self):
        self.parameterNode = slicer.vtkMRMLScriptedModuleNode()
        slicer.mrmlScene.AddNode(self.parameterNode)


class Needle_Iceball3Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        self.layout.addStretch()
        self.layout.setSpacing(5)
        
        # 3D Volume Selector
        self.volumeSelectorLabel = qt.QLabel("Select the binary tumor volume:")
        self.layout.addWidget(self.volumeSelectorLabel)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.setFixedHeight(30)
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Select the binary tumor volume.")
        self.layout.addWidget(self.volumeSelector)
        
        # 3D Surface Mesh Selector
        self.meshSelectorLabel = qt.QLabel("Select the needle path surface mesh:")
        self.layout.addWidget(self.meshSelectorLabel)
        
        self.meshSelector = slicer.qMRMLNodeComboBox()
        self.meshSelector.nodeTypes = ["vtkMRMLModelNode"]  # Node type for surface meshes
        self.meshSelector.selectNodeUponCreation = True
        self.meshSelector.addEnabled = False
        self.meshSelector.removeEnabled = False
        self.meshSelector.noneEnabled = False
        self.meshSelector.setMRMLScene(slicer.mrmlScene)
        self.meshSelector.setFixedHeight(30) 
        self.meshSelector.setToolTip("Select the surface mesh (vtkPolyData).")
        self.layout.addWidget(self.meshSelector)
        
        self.markupsNodeSelectorLabel = qt.QLabel("Select the intended needle placement location markups node:")
        self.layout.addWidget(self.markupsNodeSelectorLabel)
        
        self.markupsNodeSelector = slicer.qMRMLNodeComboBox()
        self.markupsNodeSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupsNodeSelector.selectNodeUponCreation = True
        self.markupsNodeSelector.addEnabled = False
        self.markupsNodeSelector.removeEnabled = False
        self.markupsNodeSelector.noneEnabled = False
        self.markupsNodeSelector.showHidden = False
        self.markupsNodeSelector.showChildNodeTypes = False
        self.markupsNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.markupsNodeSelector.setToolTip("Select a markups node for the target")
        self.layout.addWidget(self.markupsNodeSelector)
        
        #Button to select Excel file
        self.selectButton = qt.QPushButton("Select Excel File with historical miss data")
        self.selectButton.toolTip = "Choose an excel file with miss vectors to be used as confidence interval."
        self.layout.addWidget(self.selectButton)
        self.selectButton.clicked.connect(self.LoadingPoints)
        
        
        # Create a checkbox for enabling/disabling automation
        self.automationCheckbox = ctk.ctkCheckBox()
        self.automationCheckbox.text = "Determine Third target (only possible after second is determined through module)"
        self.layout.addWidget(self.automationCheckbox)
        # Connect the checkbox to a function
        self.automationCheckbox.connect('toggled(bool)', self.onAutomationToggled)        
    
        # 3D Point Selection Button for First Target
        self.firstTargetButton = qt.QPushButton("Select needle in 3D")
        self.firstTargetButton.toolTip = "Pick a point for the first (and second) needle insertion in 3D space."
        self.layout.addWidget(self.firstTargetButton)
        self.firstTargetButton.connect('clicked(bool)', self.onSelectFirstNeedle)
        
        # Calculate and Place Second Target Button
        self.secondTargetButton = qt.QPushButton("Calculate possbile consecutive needle location")
        self.secondTargetButton.toolTip = "Automatically calculate the second/third target within the volume."
        self.layout.addWidget(self.secondTargetButton)
        self.secondTargetButton.connect('clicked(bool)', self.onPlaceSecondTarget)

        # Segment urethra and Calculate 5 AI Iceball
        self.Urethra = qt.QPushButton("Segment Urethra")
        self.Urethra.toolTip = "Segment the urethra to use for the AI prediction"
        self.layout.addWidget(self.Urethra)
        self.Urethra.connect('clicked(bool)', self.onSegmentUrethra)
        
        self.inputSelectorLabel = qt.QLabel("Select volume to calculate AI iceball on")
        self.layout.addWidget(self.inputSelectorLabel)
          
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setFixedHeight(30) 
        self.inputSelector.setToolTip("Select volume to calculate AI iceball with.")
        self.layout.addWidget(self.inputSelector)
        
        self.outputSelectorLabel = qt.QLabel("Select output volume (None)")
        self.layout.addWidget(self.outputSelectorLabel)
        
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]  # Set node type for output
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True  # Allow the creation of a new output volume
        self.outputSelector.removeEnabled = False
        self.outputSelector.noneEnabled = True  # Allow "None" selection
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setFixedHeight(30)
        self.outputSelector.setToolTip("Select or create output volume (None for no output).")
        self.layout.addWidget(self.outputSelector)
         
        self.Best = qt.QPushButton("Give me the best point")
        self.Best.toolTip = "Calculate the best AI prediction"
        self.layout.addWidget(self.Best)
        self.Best.connect('clicked(bool)', self.onBest)

        
        self.timer = QTimer()
        self.startTime = None
         # Set the timer interval (milliseconds)
        
        # Initialize variables
        self.firstTargetRAS = None
        self.volumeNode = None
        self.placement = 'Second placement'
        self.top5_points = None
        self.logic = PredictionLogic()

#%%
    """
    Part to select first needle and save it in a markuspnode with name 'Needle'
    """
        
    def onSelectFirstNeedle(self):
        self.volumeNode = self.volumeSelector.currentNode()
        if not self.volumeNode:
            slicer.util.errorDisplay("Please select a volume first.")
            return
        
        volumeArray = slicer.util.arrayFromVolume(self.volumeNode)
        uniqueValues = np.unique(volumeArray)
        
        if len(uniqueValues) > 2 or not np.array_equal(uniqueValues, [0, 1]):
            slicer.util.errorDisplay("Selected volume is not binary. Please select a binary volume (only 0 and 1 values).")
            return
        
        self.placeFirstNeedle()
        markupsNodeTarget = slicer.util.getNode('Needle')   
        markupsNodeTarget.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onNeedlePointAdded)
        
        
    def placeFirstNeedle(self):
        # Place the first needle in 3D space
        self.markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        self.markupsNode.SetName("Needle")
        slicer.modules.markups.logic().StartPlaceMode(1)
        
    def onNeedlePointAdded(self, caller, event):
        # Trigger stop selecting when the first point is added
        if self.placement == 'Second placement':
            if caller.GetNumberOfControlPoints() > 1:
                self.onStopSelecting()
        if self.placement == 'Third placement':
            if caller.GetNumberOfControlPoints() > 2:
                self.onStopSelecting()
        self.firstTargetButton.setStyleSheet("background-color: green; color: white;")       
#%%
    def LoadingPoints(self):
        """
        Loads historical data and saves it as an array
        """
        # Open a file dialog to select an Excel file
        self.showPopupMessageExcel()
        
        fileDialog = qt.QFileDialog()
        fileDialog.setNameFilter("Excel Files (*.xlsx *.xls)")
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)

        if fileDialog.exec_():
            selectedFiles = fileDialog.selectedFiles()
            filePath = selectedFiles[0]

        data = pd.read_excel(filePath)
       

        R_coords = data.iloc[:, 6]
        A_coords = data.iloc[:, 7]
        S_coords = data.iloc[:, 8]

        self.points = np.array(list(zip(R_coords,A_coords,S_coords)))
        self.selectButton.setStyleSheet("background-color: green; color: white;")
#%%
    def onAutomationToggled(self, checked):
        """
        Checks if there are two or three needles used
        """
        if checked:
            # Automation enabled logic
            print("Third placement")
            self.placement = 'Third placement'
            self.firstTargetButton.setStyleSheet("")
            self.secondTargetButton.setStyleSheet("")
            self.Best.setStyleSheet("")
        else:
            # Automation disabled logic
            print("Second placement")  
            self.placement = 'Second placement'
        
    def onStopSelecting(self):
        """
        Used to stop selecting a point
        """
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
        slicer.util.infoDisplay("Point placement mode stopped.")
#%%
    """
    Part used to time a certain action
    """
    def startTimer(self):
        self.timer_interval = 5000 
        self.start_time = time.time()  # Record the start time
        self.timer.start(self.timer_interval)  # Start the timer

    def stopTimer(self):
        self.timer.stop()  # Stop the timer
        self.elapsed_time = time.time() - self.start_time
    def onTimerTimeout(self):
        # This method will be called every 'self.timer_interval' milliseconds
        slicer.util.messageBox("Timer triggered!")  # Example action
        
#%%
    def showPopupMessageExcel(self):
        """
        Used to let the user know that a certain task is being performed
        """
        # Create a message box
        msgBox = qt.QMessageBox()
    
        # Set the message box's content
        msgBox.setIcon(qt.QMessageBox.Information)  # Type of icon (Information, Warning, Critical, etc.)
        msgBox.setText("Excel sheet format")
        msgBox.setInformativeText("The excel sheet should have R, A, S, deviation lengths(mm) per insertion in columns G,H,I")
        msgBox.setWindowTitle("Select your excel sheet")
        
        # Add buttons (optional)
        msgBox.setStandardButtons(qt.QMessageBox.Ok)  # Add an OK button
    
        # Show the message box (exec() makes it modal)
        msgBox.exec()
        
    def showPopupMessage(self):
        """
        Used to let the user know that a certain task is being performed
        """
        # Create a message box
        msgBox = qt.QMessageBox()
    
        # Set the message box's content
        msgBox.setIcon(qt.QMessageBox.Information)  # Type of icon (Information, Warning, Critical, etc.)
        msgBox.setText("A task is being performed")
        msgBox.setInformativeText("Please wait until the task completes.")
        msgBox.setWindowTitle("Task in Progress")
        
        # Add buttons (optional)
        msgBox.setStandardButtons(qt.QMessageBox.Ok)  # Add an OK button
    
        # Show the message box (exec() makes it modal)
        msgBox.exec()
        
    def showPopupMessage2(self, text):
        """
        Used to let the user know that a certain task is being performed
        """
        # Create a message box
        msgBox = qt.QMessageBox()
    
        # Set the message box's content
        msgBox.setIcon(qt.QMessageBox.Information)  # Type of icon (Information, Warning, Critical, etc.)
        msgBox.setText(text)
        msgBox.setWindowTitle("First Needle placement")
        
        # Add buttons (optional)
        msgBox.setStandardButtons(qt.QMessageBox.Ok)  # Add an OK button
    
        # Show the message box (exec() makes it modal)
        msgBox.exec()        
#%%     
    def onPlaceSecondTarget(self):
        """
        Calculates center, radius and eigenvector from historical miss data with function calculate_ellipsoid_properties
        Calculates points and fractions with function calculateSecondTarget
        Visualizes heatmap
        """
        self.showPopupMessage()
        self.startTimer()
        self.center, self.radii, self.eigenvector= self.calculate_ellipsoid_properties(self.points)
        points, fractions = self.calculateSecondTarget(self.volumeNode)
    
        
#        self.visualizeTargets(self.volumeNode, points, fractions) #creates heatmap
        self.stopTimer()
        slicer.util.messageBox(f"Processing completed in {self.elapsed_time:.2f} seconds.")
        self.secondTargetButton.setStyleSheet("background-color: green; color: white;")
#%%        
    def calculate_ellipsoid_properties(self, points, confidence_level=0.95):
        """
        Calculates center, radius and eigenvectors of a confidence ellipse based 
        on a set of points
        
        """
        # 1. Calculate the center (mean) of the data points     
        center = np.mean(points, axis=0)   
        # 2. Calculate the covariance matrix of the points
        cov_matrix = np.cov(points.T) 
        # 3. Perform eigenvalue decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  
        # 4. Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort indices in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]  # Columns of eigenvectors are sorted

        # 4. Scale the axes to the specified confidence level (default is 95%)
        chi2_val = chi2.ppf(confidence_level, df=3)  # Confidence level in 3D   
        # The lengths of the ellipsoid axes are proportional to sqrt(eigenvalue * chi2_val)
        radii = np.sqrt(eigenvalues * chi2_val)

        return center, radii, eigenvectors
        
    def calculateSecondTarget(self, volumeNode):
        """
        1. Creates a set of points 1.5 cm from first needle insertion
        2. Calculates voxels for general 95% confidence ellipse in the direction of the first miss with function voxelize_ellipsoid
        3. For each point, recenters the 95% confidence ellipse to its point and calculates overlap between ellipse and tumor with function ellipsoid_and_tumor_overlap_percentage
        3. For each point, calculates iceball formation with function voxelize_ellipsoid, and calculates overlap  between ellipse and tumor with function ellipsoid_and_tumor_overlap_percentage
        4. Optional could visualize the formed 95% confidence interval and iceball
        5. Normalizes both overlap scores
        6. Combines 95% confidence and iceball with tumor overlap scores into one final value, where weighting factors W1 and W2 could be adjusted
        """
        #determine first needle location
        markupsNodeTarget = slicer.util.getNode('Needle')
        self.firstNeedle = [0.0, 0.0, 0.0]
        markupsNodeTarget.GetNthControlPointPosition(0, self.firstNeedle)
        
        
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        volumeArray = (volumeArray > 0).astype(np.int32)
        TensorVolume = torch.tensor(volumeArray, device='cuda')

        w1 = 0.5 #weight for confidence ellipse and tumor overlap
        w2 = 0.5 #weight for iceball and tumor overlap
        overlapT = []
        overlapI = []
       
    
        if self.placement == 'Second placement':
            self.random_points2 = self.generateRandomPoint_pairs_InVolume(self.firstNeedle, radius=10, num_pairs=12)
            self.random_points = self.create_points_in_mesh(distance_threshold_lower= 7.5, distance_threshold_upper=22.5)

            #self.visualizeRandomPoint_pairs(self.random_points)
           # self.random_points = self.random_points[316:]       
            self.directionvector1 = self.calculate_direction_miss(0)
            self.directionvector2 = None
            directionvectorTotal = self.total_direction_vector() #calculate direction vector of the miss

            for p in range(len(self.random_points)):
             
                visual_ellipse, ellipsoid_voxels= self.voxelize_ellipsoid(self.radii, volumeNode, directionvectorTotal) #calculate 95% confidence ellipse
                overlapT_Ellipse1 = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, self.random_points[p][1], ellipsoid_voxels)
                overlapT_Ellipse2 = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, self.random_points[p][2], ellipsoid_voxels)
                overlapT.append((overlapT_Ellipse1+overlapT_Ellipse2)/2)
                
                random_points0 = [self.random_points[p][0][0] ,self.random_points[p][0][1], self.random_points[p][0][2] - 0.5*10]
                random_points1 = [self.random_points[p][1][0] ,self.random_points[p][1][1], self.random_points[p][1][2] - 0.5*10]
                random_points2 = [self.random_points[p][2][0] ,self.random_points[p][2][1], self.random_points[p][2][2] - 0.5*10]
                overlapiceball_tumor = self.three_voxelized_ellipsoids(volumeNode, TensorVolume, random_points0,random_points1, random_points2)
                overlapI.append(overlapiceball_tumor)
            
        if self.placement == 'Third placement':

            # Define the name of the segmentation nodes to hide
            segmentation_name = "GeometricIceball"
            
            # Iterate over all segmentation nodes in the scene
            segmentation_nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
            for i in range(segmentation_nodes.GetNumberOfItems()):
                segmentationNode = segmentation_nodes.GetItemAsObject(i)
                if segmentationNode.GetName() == segmentation_name:  # Match the name
                    segmentation = segmentationNode.GetSegmentation()
                    
                    # Ensure the display node exists
                    if not segmentationNode.GetDisplayNode():
                        segmentationNode.CreateDefaultDisplayNodes()
                    
                    # Hide all segments in the segmentation node
                    for j in range(segmentation.GetNumberOfSegments()):
                        segmentId = segmentation.GetNthSegmentID(j)
                        segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentId, False)
                    
            # Construct the segmentation node name
            markupNodeName = ['5 Best points', 'Best Points', 'Best AI generated Point']
            for markupsNodeName in markupNodeName:        
                try:
                    # Retrieve the markups node by its name
                    markupsNode = slicer.util.getNode(markupsNodeName)
                except slicer.util.MRMLNodeNotFoundException:
              
                    continue  # Skip to the next node if not found
            
                # Get the display node of the markups node
                displayNode = markupsNode.GetDisplayNode()
                if displayNode:
                    # Set visibility to false to hide it in the 3D and 2D views
                    displayNode.SetVisibility(False)
                    displayNode.SetVisibility2D(False)  # Optional: hide in slice views as well
                
             
            markupsNodeTarget = slicer.util.getNode('Needle')
            self.SecondNeedle = [0.0, 0.0, 0.0]
            if markupsNodeTarget.GetNthControlPointPosition(1, self.SecondNeedle) == None:
                print('First select the first and second needle')
            else:
                markupsNodeTarget.GetNthControlPointPosition(1, self.SecondNeedle)
            self.random_points = self.create_third_point()
            
            #self.visualizeRandomPoint_pairs(self.random_points)
          
            self.directionvector1 = self.calculate_direction_miss(0)
            self.directionvector2 = self.calculate_direction_miss(1)
            directionvectorTotal = self.total_direction_vector() #calculate direction vector of the miss
            for p in range(len(self.random_points)):              
                visual_ellipse, ellipsoid_voxels= self.voxelize_ellipsoid(self.radii, volumeNode, directionvectorTotal) #calculate 95% confidence ellipse
                overlapT_Ellipse3 = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, self.random_points[p][2], ellipsoid_voxels)
                overlapT.append(overlapT_Ellipse3)
                
                random_points0 = [self.random_points[p][0][0] ,self.random_points[p][0][1], self.random_points[p][0][2] - 0.5*10]
                random_points1 = [self.random_points[p][1][0] ,self.random_points[p][1][1], self.random_points[p][1][2] - 0.5*10]
                random_points2 = [self.random_points[p][2][0] ,self.random_points[p][2][1], self.random_points[p][2][2] - 0.5*10]
                overlapiceball_tumor = self.three_voxelized_ellipsoids(volumeNode, TensorVolume, random_points0,random_points1, random_points2)
                overlapI.append(overlapiceball_tumor)



        #Normalize fractions
        if max(overlapT) == min(overlapT):
            # If they are, set all normalized values to 0 (or another fixed value)
            normalized_set1 = [0 for _ in overlapT]
        else:
            normalized_set1 = [(f - min(overlapT)) / (max(overlapT) - min(overlapT)) for f in overlapT]
            
        if max(overlapI) == min(overlapI):
            # If they are, set all normalized values to 0 (or another fixed value)
            normalized_set2 = [0 for _ in overlapI]
        else:
            normalized_set2 = [(f - min(overlapI)) / (max(overlapI) - min(overlapI)) for f in overlapI]
        
        normalized_combined = [w1* f1 + w2*f2  for f1, f2 in zip(normalized_set1, normalized_set2)]
        #normalized_combined = [(f - min(combined_fractions)) / (max(combined_fractions) - min(combined_fractions)) for f in combined_fractions]
        
#        self.save_list_to_excel(overlapI, 'Iceball.xlsx')
#        self.save_list_to_excel(overlapT, 'confidence.xlsx')
#        self.save_list_to_excel(combined_fractions, 'combined.xlsx')

        index = normalized_combined.index(max(normalized_combined))
     
       
        top_5index =np.argsort(normalized_combined)[-5:][::-1]
     
        self.top5_points = [self.random_points[i] for i in top_5index]    
        
        self.visualize_BestRandomPoint_pairs(self.top5_points)
        for i in range(len(self.top5_points)):
            name = 'Point Option' + str(i)
            BestPoint1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            BestPoint1.SetName(name)
            BestPoint1.AddControlPoint(self.top5_points[i][0])
            BestPoint1.AddControlPoint(self.top5_points[i][1]) 
            BestPoint1.AddControlPoint(self.top5_points[i][2]) 
            cluster1Display = BestPoint1.GetDisplayNode()
            cluster1Display.SetTextScale(0)  # Hide the label text
            cluster1Display.SetGlyphScale(0)  # Adjust point size
            cluster1Display.SetSelectedColor(0,1, 0)  # Red for cluster 1
        print('Yellow point for best geometrical location')
        print('White (hidden) points for second best geometrical locations, also used during AI calculations')           
            
        #self.visualize_BestRandomPoint_pairs(self.top5_points)
#        randomPointsNode = slicer.util.getNode("Best Points") 
#        self.focusSlicesOnPoints(randomPointsNode)

#        if self.placement == 'Second placement':
#            self.visualize_ellipsoid(volumeNode, visual_ellipse, self.firstNeedle)
#            self.visualize_ellipsoid(volumeNode, visual_ellipse, self.random_points[index][2])
#        if self.placement == 'Third placement':        
#            self.visualize_ellipsoid(volumeNode, visual_ellipse, random_points[index][2])

        random_points0 = [self.random_points[index][0][0] ,self.random_points[index][0][1], self.random_points[index][0][2] - 0.5*10]
        random_points1 = [self.random_points[index][1][0] ,self.random_points[index][1][1], self.random_points[index][1][2] - 0.5*10]
        random_points2 = [self.random_points[index][2][0] ,self.random_points[index][2][1], self.random_points[index][2][2] - 0.5*10]

        self.visualize_three_voxelized(volumeNode, random_points0,random_points1, random_points2)
        
        markupsNode = slicer.util.getNode("Best Points")
        self.focusSlicesOnPoints(markupsNode)      
                
        return self.random_points, normalized_combined
    
#%%
    """Functions for second placement"""
    def generateRandomPoint_pairs_InVolume(self, centerPointRAS, radius, num_pairs):
        """
        Generate symmetric pairs of points on the circumference of a circle in the XY plane.
        Each pair is positioned symmetrically around a central point with a 60-degree angle between them.
        
        Parameters:
            centerPointRAS: Tuple of (x, y, z) representing the center of the circle in RAS coordinates.
            radius: The radius of the circle in millimeters.
            num_pairs: The number of symmetric pairs to generate along the circumference.
        
        Returns:
            A list of tuples, where each tuple contains two points (each point is a numpy array of [x, y, z]),
            arranged symmetrically around each center point.
        """
        # List to store the point pairs
        point_pairs = []
        
        # Angle increment for each central point to be evenly distributed
        angle_increment = 2 * np.pi / num_pairs
        offset_angle = np.pi / 6  # Offset of 30 degrees for each side point in the pair
        # Z-offsets for each circle layer relative to the center Z coordinate
        distances = [-6, -4, -2, 0, 2, 4, 6]
        
        for z_offset in distances:
            # Adjust centerPointRAS's z-coordinate for the current layer
            adjusted_center_z = centerPointRAS[2] + z_offset
            for i in range(num_pairs):
                # Calculate the angle for the central point
                angle_center = i * angle_increment
                
                # Calculate coordinates of the left point (center - 30 degrees)
                angle_left = angle_center - offset_angle
                x_left = centerPointRAS[0] + radius * np.cos(angle_left)
                y_left = centerPointRAS[1] + radius * np.sin(angle_left)
                z_left = adjusted_center_z
                point_left = np.array([x_left, y_left, z_left])
                
                # Calculate coordinates of the right point (center + 30 degrees)
                angle_right = angle_center + offset_angle
                x_right = centerPointRAS[0] + radius * np.cos(angle_right)
                y_right = centerPointRAS[1] + radius * np.sin(angle_right)
                z_right = adjusted_center_z
                point_right = np.array([x_right, y_right, z_right])
                center = [centerPointRAS[0], centerPointRAS[1], adjusted_center_z]
                # Add the symmetric pair to the list
                point_pairs.append((center, point_left, point_right))
        
        return point_pairs

    def create_points_in_mesh(self, distance_threshold_lower= 7.5, distance_threshold_upper=22.5): 
        """Function to generate points at certain gridholes, taking into account the needle template"""
        selectedMeshNode = self.meshSelector.currentNode()
        if not selectedMeshNode:
            print("No mesh node selected.")
            return
    
        surfaceMesh = selectedMeshNode.GetPolyData()
        if not surfaceMesh:
            print("Surface mesh is invalid.")
            return
    
        # Step 1: Segment the mesh into cylinders
        cylinderMeshes = self.segment_mesh_into_cylinders(surfaceMesh)
        cylinder_centers, axes = self.process_cylinders(cylinderMeshes)
       
        # Find the closest cylinder to the first needle point
        closest_center_index, closest_center, first_direction = self.find_closest_cylinder(self.firstNeedle, cylinder_centers, axes)
       
        centers_within_distance = self.find_centers_within_distance(closest_center, cylinder_centers, axes, distance_threshold_lower, distance_threshold_upper)
    
        
        point_duo = []
        target_height = self.firstNeedle[2]
        depths = [target_height - 15,target_height - 10,target_height - 5, target_height, target_height + 5,target_height +10,target_height + 15]  # Voorbeeld dieptes (10 mm onder, op, en boven de eerste hoogte)
        #depths = [target_height]
        for depth in depths:  
            points_onmesh = []
            firstNeedle = []
            for i, cylinder in enumerate(centers_within_distance):
                center = cylinder['center']
                direction = cylinder['axis']
             
                needle_point, cylinder_point = self.generate_point_at_depth(center, direction, self.firstNeedle, first_direction, depth)   
                points_onmesh.append(cylinder_point) 
                firstNeedle.append(needle_point)
      
            duo = self.generate_point_duos(points_onmesh, firstNeedle)
            
            point_duo = point_duo+duo

        return point_duo

    def find_centers_within_distance(self, closest_center, cylinder_centers, axes, distance_threshold_lower= 7.5, distance_threshold_upper=22.5):
        """
        Find cylinder centers and their axes that are within a specified distance (in mm) from the closest center.
        
        :param closest_center: The coordinates of the closest cylinder center.
        :param cylinder_centers: List of cylinder centers (each center is an array [x, y, z]).
        :param axes: List of direction vectors for the cylinders (each axis is an array [ax, ay, az]).
        :param distance_threshold: The maximum distance in mm.
        :return: List of dictionaries with 'center' and 'axis' for cylinders within the specified distance.
        """
        # List to store centers and axes within the distance threshold
        centers_within_distance = []
        
        # Iterate over all cylinder centers and calculate the distance
        for i, center in enumerate(cylinder_centers):
            axis = axes[i]
            
            # Calculate the Euclidean distance between the closest center and the current center
            distance = euclidean(closest_center, center)
            
            # If the distance is within the threshold, add the center and axis to the list
            if distance_threshold_lower < distance <= distance_threshold_upper:
                centers_within_distance.append({'center': center, 'axis': axis})
        
        return centers_within_distance
    
    def generate_point_duos(self, points, first_needle,  min_dist = 7.5, max_dist=22.5):
        """
        Generate all duos of points that are within a given distance.
        
        :param points: List of points, each as [x, y, z].
        :param max_dist: Maximum distance between points to form a valid duo.
        :return: List of duos, where each duo is a tuple of two points ([p1], [p2]).
        """
        valid_duos = []
        n = len(points)
    
        for i in range(n):
            for j in range(i + 1, n):  # Avoid duplicate pairs and self-pairs
                if min_dist < euclidean(points[i], points[j]) <= max_dist:
                    valid_duos.append((first_needle[i], points[i], points[j]))
  
        return valid_duos
 
#%%
    """Functions for third placement"""
    def generateRandomPoint3(self, centerPointRAS1, centerPointRAS2):
        """
        Generate evenly distributed points on multiple circular layers around two given points,
        ensuring the third point is exactly 1 cm away from both points.
    
        Parameters:
            centerPointRAS1: The coordinates of the first point in RAS (x, y, z).
            centerPointRAS2: The coordinates of the second point in RAS (x, y, z).
    
        Returns:
            A list of points for each layer (in RAS coordinates).
        """
        # List to store all points on each circular layer
        points = []
    
        # Z-offsets for each circle layer relative to the center Z coordinate
        distances = [-6, -4, -2, 0, 2, 4, 6]
    
        p1 = np.array(centerPointRAS1)
        p2 = np.array(centerPointRAS2)
    
        # Direction vector and distance between points
        d = p2 - p1
        distance = np.linalg.norm(d)
    
        if distance > 20.0:
            raise ValueError("The two points are too far apart to find a third point that is 1 cm from both.")
    
        # Midpoint of the line segment
        M = (p1 + p2) / 2
    
        # Height of the equilateral triangle
        h = np.sqrt(10**2 - (distance / 2)**2)
    
        # Normal vector to the line segment
        if np.allclose(d[:2], 0):  # Handle edge case where d lies along the z-axis
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 0, 1])  # Arbitrary vector not parallel to d
        n = np.cross(d, v)
        n = n / np.linalg.norm(n)  # Normalize
    
        # Two possible solutions for the third point
        p3_1 = M + h * n
        p3_2 = M - h * n
    
        # Loop through each z-offset to create a new circle layer at that height
        for z_offset in distances:
            # Adjust z-coordinates for the current layer
            adjusted_center_z1 = centerPointRAS1[2] + z_offset
            adjusted_center_z2 = centerPointRAS2[2] + z_offset
            PointRas1 = (centerPointRAS1[0], centerPointRAS1[1], adjusted_center_z1)
            PointRas2 = (centerPointRAS2[0], centerPointRAS2[1], adjusted_center_z2)
            PointRas3_1 = (p3_1[0], p3_1[1], p3_1[2] + z_offset)
            PointRas3_2 = (p3_2[0], p3_2[1], p3_2[2] + z_offset)
    
            points.append((PointRas1, PointRas2, PointRas3_1))
            points.append((PointRas1, PointRas2, PointRas3_2))
    
        return points
   
    def three_voxelized_ellipsoids(self, volumeNode,mask_tensor, position1, position2, position3):
        radii = (12,12,15)
        direction_vector = np.array([1,0,0])
        ellipsoid_voxels,nee = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)        
#        self.visualize_ellipsoid(volumeNode, nee, position1)
#        self.visualize_ellipsoid(volumeNode, nee, position2)
#        self.visualize_ellipsoid(volumeNode, nee, position3)
        overlap = self.ellipsoid_and_tumor_overlap_percentage2(volumeNode, mask_tensor, position1, position2, position3, ellipsoid_voxels)

        return overlap
    
    def ellipsoid_and_tumor_overlap_percentage2(self, volumeNode, mask_tensor, position1, position2, position3, ellipsoid_voxels):
        """
        Compares the voxels from the ellipsoid with the tumor mask, to define the overlap score
        """
        # Ensure binary mask tensor on GPU
        mask_tensor = (mask_tensor > 0).int().to('cuda')
        
        # Convert ellipsoid center point to IJK and apply voxel scaling
        Ijkpoint = self.rasToIJK(volumeNode, position1)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels1 = torch.round(shifted_ellipsoid_voxels).long()
        
        Ijkpoint = self.rasToIJK(volumeNode, position2)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels2 = torch.round(shifted_ellipsoid_voxels).long()

        Ijkpoint = self.rasToIJK(volumeNode, position3)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels3 = torch.round(shifted_ellipsoid_voxels).long()        

        combined_ellipsoid = torch.cat([shifted_ellipsoid_voxels1, shifted_ellipsoid_voxels2, shifted_ellipsoid_voxels3], dim=0)

        # In-bounds filtering for shifted ellipsoid voxels
        in_bounds = (
            (combined_ellipsoid[:, 0] >= 0) & 
            (combined_ellipsoid[:, 0] < mask_tensor.shape[0]) & 
            (combined_ellipsoid[:, 1] >= 0) & 
            (combined_ellipsoid[:, 1] < mask_tensor.shape[1]) & 
            (combined_ellipsoid[:, 2] >= 0) & 
            (combined_ellipsoid[:, 2] < mask_tensor.shape[2])
        )
        valid_ellipsoid_voxels = combined_ellipsoid[in_bounds].long()
     
        # Count voxels within tumor mask and ellipsoid
        inside_mask_count = (mask_tensor[valid_ellipsoid_voxels[:, 0], valid_ellipsoid_voxels[:, 1], valid_ellipsoid_voxels[:, 2]]>0).sum().item()

        # Tumor Size for overlap calculation
        TumorSize = mask_tensor.sum().item()
        
        # Fraction of ellipsoid voxels within tumor mask
        fraction_in_tumor = (inside_mask_count / TumorSize) if TumorSize > 0 else 0.0

        return fraction_in_tumor
    
    def create_third_point(self, distance_threshold_lower= 7.5, distance_threshold_upper=22.5):
        selectedMeshNode = self.meshSelector.currentNode()
        if not selectedMeshNode:
            print("No mesh node selected.")
            return
    
        surfaceMesh = selectedMeshNode.GetPolyData()
        if not surfaceMesh:
            print("Surface mesh is invalid.")
            return
    
        # Step 1: Segment the mesh into cylinders
        cylinderMeshes = self.segment_mesh_into_cylinders(surfaceMesh)
        cylinder_centers, axes = self.process_cylinders(cylinderMeshes)
       
        # Find the closest cylinder to the first needle point
        closest_center_index1, closest_center1, first_direction = self.find_closest_cylinder(self.firstNeedle, cylinder_centers, axes)  
        closest_center_index2, closest_center2, second_direction = self.find_closest_cylinder(self.SecondNeedle, cylinder_centers, axes)  
        
        centers_within_distance = self.find_centers_within_distance3(closest_center1, closest_center2, cylinder_centers, axes, distance_threshold_lower= 7.5, distance_threshold_upper=22.5)
        point_trios = []
        target_height = self.firstNeedle[2]
        depths = [target_height - 15,target_height - 10,target_height - 5, target_height, target_height + 5,target_height +10,target_height + 15]  # Voorbeeld dieptes (10 mm onder, op, en boven de eerste hoogte)
        #depths = [target_height]
        for depth in depths:  
            for i, cylinder in enumerate(centers_within_distance):
                center = cylinder['center']
                direction = cylinder['axis']
                needle_point1, needle_point2, cylinder_point = self.generate_point_at_depth3(center, direction, self.firstNeedle, first_direction,self.SecondNeedle,second_direction, depth)
                
                point_trios.append((needle_point1, needle_point2, cylinder_point))

        return point_trios

    def find_centers_within_distance3(self, closest_center1, closest_center2, cylinder_centers, axes, distance_threshold_lower= 7.5, distance_threshold_upper=22.5):
        """
        Find cylinder centers and their axes that are within a specified distance (in mm) from the closest center.
        
        :param closest_center: The coordinates of the closest cylinder center.
        :param cylinder_centers: List of cylinder centers (each center is an array [x, y, z]).
        :param axes: List of direction vectors for the cylinders (each axis is an array [ax, ay, az]).
        :param distance_threshold: The maximum distance in mm.
        :return: List of dictionaries with 'center' and 'axis' for cylinders within the specified distance.
        """
        # List to store centers and axes within the distance threshold
        centers_within_distance = []
        
        # Iterate over all cylinder centers and calculate the distance
        for i, center in enumerate(cylinder_centers):
            axis = axes[i]
            
            # Calculate the Euclidean distance between the closest center and the current center
            distance1 = euclidean(closest_center1, center)
            distance2 = euclidean(closest_center2, center)
            
            # If the distance is within the threshold, add the center and axis to the list
            if distance_threshold_lower < distance1 <= distance_threshold_upper and distance_threshold_lower < distance2 <= distance_threshold_upper:
                centers_within_distance.append({'center': center, 'axis': axis})
        
        return centers_within_distance
    
    def generate_point_at_depth3(self, center, direction, first_needle, first_direction,second_needle,second_direction, depth):
        """
        Generate points at different depths based on the first needle height and the cylinder axis.
        
        :param center: The center of the cylinder [x, y, z].
        :param direction: The direction vector of the cylinder axis [dx, dy, dz].
        :param first_needle: The position of the first needle [x, y, z].
        :param depths: List of depths relative to the first needle height.
        :return: A list of points: the first needle's point at the depths, and other points along the cylinder.
        """
        # Get the height of the first needle (z-coordinate)

        
        # The new height for the first needle based on the depth
        target_height = depth
        
        # Generate the point for the first needle at this depth
        needle_point1 = self.generate_point_at_height(first_needle, first_direction, target_height)
        
        needle_point2 = self.generate_point_at_height(second_needle, second_direction, target_height)
        
        # Generate the corresponding point along the cylinder's axis at the same depth
        cylinder_point = self.generate_point_at_height(center, direction, target_height)
     
        return needle_point1, needle_point2, cylinder_point
#%%
    """Functions to calculate direction of 95% confidence interval"""
    def calculate_direction_miss(self, number):
        """
        Used to calculate the normalized direction vector between two points,
        needle and target
        Needle: gained from manual input through onSelectFirstNeedle
        Target: gained from a markupsnode loaded into 3d slicer
        """
        markupsNodeTarget = slicer.util.getNode('Needle')
        coord1 = [0.0, 0.0, 0.0]
        markupsNodeTarget.GetNthControlPointPosition(number, coord1)
        Needle = np.array(coord1)
        
        selectedMarkupsNode = self.markupsNodeSelector.currentNode()
        
        coord2 = [0.0, 0.0, 0.0]      
        selectedMarkupsNode.GetNthControlPointPosition(number, coord2)
        Planned = np.array(coord2)
        
        difference_vector = Planned - Needle
        # Calculate the magnitude (length) of the difference vector
        magnitude = np.linalg.norm(difference_vector)
        
        # Normalize the difference vector to get the direction
        direction_vector = difference_vector / magnitude
        
        return direction_vector
    
    def total_direction_vector(self):
        """
        Calculates total direction vector if multiple are given
        """
        if self.placement == 'Second placement':
            return self.directionvector1
        if self.placement == 'Third placement':
            mean_direction = (self.directionvector1 + self.directionvector2) / 2.0
            mean_direction = mean_direction / np.linalg.norm(mean_direction)
            
            return mean_direction
#%%
    """ Functions used for second and third needle"""    
    def voxelize_ellipsoid(self, radii, volumeNode, direction_vector):
        """
        Uses the ellipsoids radius and direction vector to create a tensor voxel representation of the ellipse
        """
        spacing = volumeNode.GetSpacing()
        
        # Compute voxel radii
        rx = int(np.ceil(radii[0] / spacing[0]))  # Round up to ensure we cover the radius
        ry = int(np.ceil(radii[1] / spacing[1]))
        rz = int(np.ceil(radii[2] / spacing[2]))
        
        # Create a grid of points within the bounding box for the ellipsoid
        x = np.arange(-rx, rx + 1)
        y = np.arange(-ry, ry + 1)
        z = np.arange(-rz, rz + 1)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        
        # Use ellipsoid equation to get voxel points inside the ellipsoid
        ellipsoid_voxels = (grid_x**2 / rx**2 + grid_y**2 / ry**2 + grid_z**2 / rz**2) <= 1
        
        # Get the indices of the voxels that are inside the ellipsoid
        ellipsoid_indices = np.column_stack(np.where(ellipsoid_voxels))
        
        # Shift the indices to the center of the ellipsoid
        ellipsoid_indices -= np.array([rx, ry, rz])
        
        
        # Define the rotation matrix using the direction vector
        # Assuming that the first principal component aligns with the x-axis
        principal_axis = np.array([1, 0, 0])  # This can be any fixed axis to start with
        rotation_axis = np.cross(principal_axis, direction_vector)
        
        if np.linalg.norm(rotation_axis) > 0:  # If the rotation axis is not zero
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(principal_axis, direction_vector), -1.0, 1.0))  # Angle between the two vectors
        else:
            angle = 0  # No rotation needed if the two vectors are aligned
        
        # Create the rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle), rotation_axis[0]*rotation_axis[1]*(1 - cos_angle) - rotation_axis[2]*sin_angle, rotation_axis[0]*rotation_axis[2]*(1 - cos_angle) + rotation_axis[1]*sin_angle],
        [rotation_axis[1]*rotation_axis[0]*(1 - cos_angle) + rotation_axis[2]*sin_angle, cos_angle + rotation_axis[1]**2 * (1 - cos_angle), rotation_axis[1]*rotation_axis[2]*(1 - cos_angle) - rotation_axis[0]*sin_angle],
        [rotation_axis[2]*rotation_axis[0]*(1 - cos_angle) - rotation_axis[1]*sin_angle, rotation_axis[2]*rotation_axis[1]*(1 - cos_angle) + rotation_axis[0]*sin_angle, cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
        ])
        
        # Rotate the ellipsoid indices
        rotated_indices = ellipsoid_indices @ rotation_matrix.T  # Apply rotation
        rotated_indices_long = torch.tensor(rotated_indices, dtype=torch.long, device='cuda')
        rotated_indices_visualization = torch.tensor(rotated_indices, dtype=torch.float32, device='cuda')
        # Convert to tensor and ensure proper type
        return rotated_indices_visualization, rotated_indices_long
    
    def ellipsoid_and_tumor_overlap_percentage(self, volumeNode, mask_tensor, ellipsoid_point, ellipsoid_voxels):
        """
        Compares the voxels from the ellipsoid with the tumor mask, to define the overlap score
        """

        # Ensure binary mask tensor on GPU
        mask_tensor = (mask_tensor > 0).int().to('cuda')
        
        # Convert ellipsoid center point to IJK and apply voxel scaling
        Ijkpoint = self.rasToIJK(volumeNode, ellipsoid_point)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels = torch.round(shifted_ellipsoid_voxels).long()

        # In-bounds filtering for shifted ellipsoid voxels
        in_bounds = (
            (shifted_ellipsoid_voxels[:, 0] >= 0) & 
            (shifted_ellipsoid_voxels[:, 0] < mask_tensor.shape[0]) & 
            (shifted_ellipsoid_voxels[:, 1] >= 0) & 
            (shifted_ellipsoid_voxels[:, 1] < mask_tensor.shape[1]) & 
            (shifted_ellipsoid_voxels[:, 2] >= 0) & 
            (shifted_ellipsoid_voxels[:, 2] < mask_tensor.shape[2])
        )
        valid_ellipsoid_voxels = shifted_ellipsoid_voxels[in_bounds].long()

        # Count voxels within tumor mask and ellipsoid
        inside_mask_count = (mask_tensor[valid_ellipsoid_voxels[:, 0], valid_ellipsoid_voxels[:, 1], valid_ellipsoid_voxels[:, 2]]>0).sum().item()

        # Tumor Size for overlap calculation
        TumorSize = mask_tensor.sum().item()
        
        # Fraction of ellipsoid voxels within tumor mask
        fraction_in_tumor = (inside_mask_count / TumorSize) if TumorSize > 0 else 0.0

        return fraction_in_tumor
      
    def rasToIJK(self, volumeNode, ras_point):
        # Get spacing, origin, and direction matrix
        spacing = volumeNode.GetSpacing()  # (0.9375, 0.9375, 3.6)
        origin = volumeNode.GetOrigin()    # (90.583, 117.562, 8.79922)
        directionMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASDirectionMatrix(directionMatrix)

        # Subtract the origin from the RAS point to account for the translation
        ras_point_adjusted = np.array(ras_point) - np.array(origin)
    
        # Get the inverse of the direction matrix for transforming RAS to IJK
        directionMatrix.Invert()
    
        # Apply the direction matrix
        ras_homogeneous = np.array([ras_point_adjusted[0], ras_point_adjusted[1], ras_point_adjusted[2], 1])
        ijk_homogeneous = directionMatrix.MultiplyPoint(ras_homogeneous)
    
        # Divide by spacing to account for non-uniform voxel size
        ijk_coords = np.array([ijk_homogeneous[2] / spacing[2],
                               ijk_homogeneous[1] / spacing[1],
                               ijk_homogeneous[0] / spacing[0]])
    
        # Round to get integer IJK coordinates
        ijk_coords = np.round(ijk_coords).astype(int)
    
        return ijk_coords

  
    def generate_point_at_depth(self, center, direction, first_needle, first_direction, depth):
        """
        Generate points at different depths based on the first needle height and the cylinder axis.
        
        :param center: The center of the cylinder [x, y, z].
        :param direction: The direction vector of the cylinder axis [dx, dy, dz].
        :param first_needle: The position of the first needle [x, y, z].
        :param depths: List of depths relative to the first needle height.
        :return: A list of points: the first needle's point at the depths, and other points along the cylinder.
        """
        # Get the height of the first needle (z-coordinate)

        
        # The new height for the first needle based on the depth
        target_height = depth
        
        # Generate the point for the first needle at this depth
        needle_point = self.generate_point_at_height(first_needle, first_direction, target_height)
        
        # Generate the corresponding point along the cylinder's axis at the same depth
        cylinder_point = self.generate_point_at_height(center, direction, target_height)
     
        return needle_point, cylinder_point

    def generate_point_at_height(self, center, direction, target_height):
        """
        Genereer een punt op de lijn gedefinieerd door center en direction, op een specifieke hoogte.
        
        :param center: Het centrum van de lijn [x, y, z].
        :param direction: De richtingsvector van de lijn [dx, dy, dz].
        :param target_height: De z-hoogte waarop het punt moet liggen.
        :return: Het punt op de lijn op de opgegeven hoogte [x, y, z].
        """
        # Zet de richting om naar een eenheidsvector (normaaliseer de richting)
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        # Het huidige z-coördinaat van het centrum
        center_z = center[2]
        
        # Bereken de offset in de z-richting om de lijn naar de gewenste hoogte te bewegen
        z_offset = target_height - center_z
        
        # Beweeg langs de lijn om het gewenste punt te bereiken
        # De richting vector heeft een z-component die bepaalt hoe we de lijn moeten bewegen
        # Beweeg langs de lijn zodat de z-coördinaat overeenkomt met target_height
        
        scale_factor = z_offset / direction[2]  # Bepaal hoeveel we moeten schalen in de richting van de z-as
        
        # Genereer het punt door het centrum te verplaatsen langs de richting
        point_at_height = np.array(center) + scale_factor * direction
        
        return point_at_height.tolist()
        
 
    def process_cylinders(self, cylinderMeshes):
        """
        Process each segmented cylinder to compute its center and axis.
        :param cylinderMeshes: List of vtkPolyData, each representing a cylinder.
        :return: List of cylinder centers and axes.
        """
        centers = []
        axes = []
    
        for i, cylinderMesh in enumerate(cylinderMeshes):
            # Convert vtkPoints to numpy array
            pointsData = cylinderMesh.GetPoints()
            points_array = np.array([pointsData.GetPoint(j) for j in range(pointsData.GetNumberOfPoints())])
    
            # Ensure enough points are present for PCA
            if len(points_array) < 3:
                print(f"Skipping cylinder {i+1}: Not enough points for PCA.")
                continue
    
            # Calculate PCA for the cylinder points
            center, axis = self.get_cylinder_axis_from_points(points_array)
            centers.append(center)
            axes.append(axis)
  
    
        return centers, axes

    def get_cylinder_axis_from_points(self, points_array):
        """
        Use PCA to find the axis of a cylinder based on the points.
        :param points_array: Numpy array of points that belong to the cylinder.
        :return: The center of the points and the direction vector of the cylinder axis.
        """
        if len(points_array) < 2:
            raise ValueError("Insufficient points for PCA.")
        
        # Perform PCA
        pca = PCA(n_components=3)  # 3 components for 3D data
        pca.fit(points_array)
    
        # First principal component represents the cylinder axis
        axis_direction = pca.components_[0]
    
        # Calculate the center of the points
        center = np.mean(points_array, axis=0)
    
    
        # Ensure the axis is not degenerate
        if pca.explained_variance_ratio_[0] < 0.5:
            raise ValueError("Principal component does not represent a dominant direction.")
    
        return center, axis_direction   
    
    def segment_mesh_into_cylinders(self, surfaceMesh):
        """
        Segment the mesh into separate connected components (cylinders) based on cell connectivity,
        and visualize them for debugging.
        :param surfaceMesh: The vtkPolyData surface mesh.
        :return: List of vtkPolyData, each representing a segmented cylinder.
        """
        # Clean the mesh
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surfaceMesh)
        cleaner.SetTolerance(0.001)
        cleaner.Update()
        surfaceMesh = cleaner.GetOutput()
    
        # Smooth the mesh
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(surfaceMesh)
        smoother.SetNumberOfIterations(20)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        surfaceMesh = smoother.GetOutput()
    
        # Connectivity filter
        connectivityFilter = vtk.vtkConnectivityFilter()
        connectivityFilter.SetInputData(surfaceMesh)
        connectivityFilter.SetExtractionModeToAllRegions()
        connectivityFilter.ColorRegionsOn()
        connectivityFilter.Update()
    
        segmentedMesh = connectivityFilter.GetOutput()
        numRegions = connectivityFilter.GetNumberOfExtractedRegions()
    
    
  
        cylinderMeshes = []
        for regionId in range(numRegions):
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(segmentedMesh)
            threshold.SetLowerThreshold(regionId)
            threshold.SetUpperThreshold(regionId)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
            threshold.Update()
    
            surfaceFilter = vtk.vtkDataSetSurfaceFilter()
            surfaceFilter.SetInputData(threshold.GetOutput())
            surfaceFilter.Update()
    
            cylinderMesh = surfaceFilter.GetOutput()
            cylinderMeshes.append(cylinderMesh)
    
    
        return cylinderMeshes

    def find_closest_cylinder(self, needle_point, cylinder_centers, cylinder_axes):
        """
        Find the closest cylinder center to the given needle point.
        
        :param needle_point: Coordinates of the first needle (x, y, z).
        :param cylinder_centers: List of cylinder centers (each center is an array [x, y, z]).
        :param cylinder_axes: List of direction vectors for each cylinder (each direction is an array [dx, dy, dz]).
        :return: The index of the closest cylinder center, the center coordinates, and the direction.
        """
        # Initialize minimum distance to a very large number
        min_distance = float('inf')
        closest_center_index = -1
        closest_center = None
        closest_direction = None
        
        # Iterate over all cylinder centers and calculate the distance
        for i, (center, axis) in enumerate(zip(cylinder_centers, cylinder_axes)):
            # Calculate the Euclidean distance between the needle and the cylinder center
            distance = euclidean(needle_point, center)
            
            # If this distance is smaller than the current minimum, update the closest center
            if distance < min_distance:
                min_distance = distance
                closest_center_index = i
                closest_center = center
                closest_direction = axis
        
        return closest_center_index, closest_center, closest_direction

#%%
    """Functions for visualization"""
    def visualize_BestRandomPoint_pairs(self, random_points):
        """
        Visualizes given points
        """       
        randomPointsNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode2.SetName("5 Best points")

        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Best Points")
        

        for i in range(len(random_points)-1):            
            for point in random_points[i+1]:
                randomPointsNode2.AddControlPoint(point[0], point[1], point[2])
                
        for point in random_points[0]:
            randomPointsNode.AddControlPoint(point[0], point[1], point[2])
            
                # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(2)  # Adjust point size
        cluster1Display.SetSelectedColor(1,1, 0)  # Red for cluster 1

        # Set colors for the display nodes
        cluster1Display2 = randomPointsNode2.GetDisplayNode()
        cluster1Display2.SetTextScale(0)  # Hide the label text
        cluster1Display2.SetGlyphScale(0)  # Adjust point size
        cluster1Display2.SetSelectedColor(1,1,1)  # Red for cluster 1
        

        
    def visualizeBestAIPoint(self, random_points):
        """
        Visualizes given points
        """
        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Best AI generated Point")
        

        for point in random_points:
            randomPointsNode.AddControlPoint(point[0], point[1], point[2])  # Add each point as a control point
        # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(2)  # Adjust point size
        cluster1Display.SetSelectedColor(0,1, 1) # Red for cluster 1   
        
    def focusSlicesOnPoints(self, markupsNode):
        # Get the bounding box of the points
        bounding_box = [0.0] * 6
        markupsNode.GetRASBounds(bounding_box)
    
        # Calculate the center of the bounding box
        center = [
            (bounding_box[0] + bounding_box[1]) / 2,  # X center
            (bounding_box[2] + bounding_box[3]) / 2,  # Y center
            (bounding_box[4] + bounding_box[5]) / 2   # Z center
        ]
    
        # Adjust the slice views to center on the bounding box center
        layoutManager = slicer.app.layoutManager()
        for sliceViewName in ["Red", "Green", "Yellow"]:
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            if not sliceWidget:
                continue
            sliceLogic = sliceWidget.sliceLogic()
            sliceLogic.SetSliceOffset(center[2])  # Set the Z offset to focus on the points

        
    def visualizeRandomPoint_pairs(self, random_points):
        """
        Visualizes given points
        """
        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Random Points")
        
        randomPointsNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode2.SetName("Random Points2")
        
        randomPointsNode3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode3.SetName("Random Points3")
        
        randomPointsNode4 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode4.SetName("Random Points4")
  
        for i in range(0, len(random_points)-3, 4):   
            for point in random_points[i]:                  
                randomPointsNode.AddControlPoint(point[0], point[1], point[2])
            for point in random_points[i+1]:
                randomPointsNode2.AddControlPoint(point[0], point[1], point[2])
            for point in random_points[i+2]:
                randomPointsNode3.AddControlPoint(point[0], point[1], point[2])
            for point in random_points[i+3]:
                randomPointsNode4.AddControlPoint(point[0], point[1], point[2])
                   
        # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(1)  # Adjust point size
        cluster1Display.SetSelectedColor(0,1, 0)  # Red for cluster 1
        
                # # Set colors for the display nodes
        cluster1Display2 = randomPointsNode2.GetDisplayNode()
        cluster1Display2.SetTextScale(0)  # Hide the label text
        cluster1Display2.SetGlyphScale(1)  # Adjust point size
        cluster1Display2.SetSelectedColor(1,0, 0)  # Red for cluster 1
        
                # # Set colors for the display nodes
        cluster1Display3 = randomPointsNode3.GetDisplayNode()
        cluster1Display3.SetTextScale(0)  # Hide the label text
        cluster1Display3.SetGlyphScale(1)  # Adjust point size
        cluster1Display3.SetSelectedColor(0,0,1)  # Red for cluster 1
        
        
        cluster1Display4 = randomPointsNode4.GetDisplayNode()
        cluster1Display4.SetTextScale(0)  # Hide the label text
        cluster1Display4.SetGlyphScale(1)  # Adjust point size
        cluster1Display4.SetSelectedColor(0,1,1)  # Red for cluster 1
        
    def visualize_three_voxelized(self, volumeNode, position1, position2, position3):
        radii = (12,12,15)
        direction_vector = np.array([1,0,0])
        ellipsoid_voxels,nee = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)        
        self.visualize_ellipsoid(volumeNode, nee, position1)
        self.visualize_ellipsoid(volumeNode, nee, position2)
        self.visualize_ellipsoid(volumeNode, nee, position3)
    
    def visualize_ellipsoid(self, volumeNode, ellipsoid_voxels, center):
        """
        Visualizes an ellipse in 3D slicer
        """
        # Get voxel spacing
        spacing = volumeNode.GetSpacing()
        
        # Convert ellipsoid_voxels and center from voxel to RAS coordinates
        center_tensor = torch.tensor(center, dtype=torch.float32, device='cuda')
    
        ellipsoid_points = (ellipsoid_voxels * torch.tensor(spacing, device='cuda')) + center_tensor
    
        # Convert to numpy for visualization
        ellipse_points = ellipsoid_points.cpu().numpy()  # Convert to numpy for visualization
   
        vtk_points = vtk.vtkPoints()
        for point in ellipse_points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
        # Create a vtkPolyData object to store the points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
    
        # Create a surface from the points using Delaunay triangulation
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.Update()
        surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
        surfaceFilter.Update()
        polydata = surfaceFilter.GetOutput()
    
        # Create a model node and assign the polydata to it
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(polydata)
    
        # Create a segmentation node and import the model node into it
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentationNode)
        segmentationNode.SetName("GeometricIceball")
        # Set color and display properties
        segmentationNode.GetDisplayNode().SetColor(0, 1, 0)  # Set color to green
        segmentationNode.GetDisplayNode().SetVisibility3D(True)  # Show in 3D view
        segmentationNode.GetDisplayNode().SetOpacity(0.4)
        segmentationNode.GetDisplayNode().SetVisibility2DFill(False)  # Fill in 2D slice views
    
        # Reset the 3D view's focal point to focus on the ellipsoid
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()         

    def visualize_ellipsoid2(self, volumeNode, ellipsoid_voxels):
        """
        Visualizes an ellipse in 3D slicer
        """

    
        # Convert to numpy for visualization
        ellipse_points = ellipsoid_voxels.cpu().numpy()  # Convert to numpy for visualization
   
        vtk_points = vtk.vtkPoints()
        for point in ellipse_points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
        # Create a vtkPolyData object to store the points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
    
        # Create a surface from the points using Delaunay triangulation
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.Update()
        surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
        surfaceFilter.Update()
        polydata = surfaceFilter.GetOutput()
    
        # Create a model node and assign the polydata to it
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(polydata)
    
        # Create a segmentation node and import the model node into it
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        
        
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentationNode)
        
        # Set color and display properties
        segmentationNode.GetDisplayNode().SetColor(0, 1, 0)  # Set color to green
        segmentationNode.GetDisplayNode().SetVisibility3D(True)  # Show in 3D view
        segmentationNode.GetDisplayNode().SetOpacity(0.4)
        segmentationNode.GetDisplayNode().SetVisibility2DFill(False)  # Fill in 2D slice views
    
        # Reset the 3D view's focal point to focus on the ellipsoid
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()      
#%%
    def onSegmentUrethra(self):
        """Redirects user to Urethrasegmentation module"""
        slicer.util.selectModule("UrethraSegmentation")
        self.Urethra.setStyleSheet("background-color: green; color: white;")
        
#%%       
    def onBest(self):
        """Uses the AI model to determine best point option"""
        self.startTimer()
        
        UrethraNode = slicer.util.getNode('UrethraSegmentation')
        for i in range(5):
            markupsNode = slicer.util.getNode('Point Option' + str(i))
            self.logic.process(self.inputSelector.currentNode(), self.outputSelector.currentNode(),markupsNode,UrethraNode)
      
        overlapValues = []
        overlap = []
        ablation = []
        if self.placement == 'Second placement': 
            tumor_segmentation = self.volumeSelector.currentNode()#tumor     
            IceballVolume = slicer.util.getNode('IceballPred') #Iceball
            overlapIceball_Tumor = self.calculate_AI_Iceball_Tumor_overlap(tumor_segmentation, IceballVolume)
            ablation_zone = self.calculate_ablation_margin(tumor_segmentation, IceballVolume)
    
            
            overlapScore = overlapIceball_Tumor-ablation_zone
            overlapValues.append(overlapScore)
            overlap.append(overlapIceball_Tumor)
            ablation.append(ablation_zone)

        
            for i in range(4):
                IceballVolume = slicer.util.getNode('IceballPred_' + str(i+1)) #Iceball
                overlapIceball_Tumor = self.calculate_AI_Iceball_Tumor_overlap(tumor_segmentation, IceballVolume)
                ablation_zone = self.calculate_ablation_margin(tumor_segmentation, IceballVolume)
                overlapScore = overlapIceball_Tumor-ablation_zone
                overlapValues.append(overlapScore)
                overlap.append(overlapIceball_Tumor)
                ablation.append(ablation_zone)
                
        if self.placement == 'Third placement': 
            
            for i in range(5):
                tumor_segmentation = self.volumeSelector.currentNode()#tumor 
                IceballVolume = slicer.util.getNode('IceballPred_' + str(i+5)) #Iceball
                overlapIceball_Tumor = self.calculate_AI_Iceball_Tumor_overlap(tumor_segmentation, IceballVolume)
                ablation_zone = self.calculate_ablation_margin(tumor_segmentation, IceballVolume)
                overlapScore = overlapIceball_Tumor-ablation_zone
                overlapValues.append(overlapScore)
                overlap.append(overlapIceball_Tumor)
                ablation.append(ablation_zone)
        

        w1 = 0.5
        w2 = 0.5
        normalized_combined = [w1* f1 + w2*f2  for f1, f2 in zip(overlap, ablation)]
        
        
        maxValue = max(normalized_combined)  # Find the maximum value
            
        BestIndices = [index for index, value in enumerate(normalized_combined) if value == maxValue]  # Get all indices of the max value
        
        self.stopTimer()

        BestIndex = BestIndices[0]
        BestAIPoint = self.top5_points[BestIndex][1]

        self.visualizeBestAIPoint(self.top5_points[BestIndex])
        
        slicer.util.messageBox(f"Processing completed in {self.elapsed_time:.2f} seconds.")
        if BestAIPoint[2] - self.firstNeedle[2] == 0:
            text = 'First needle is at the correct depth'
            self.showPopupMessage2(text)
            print(text)
        if BestAIPoint[2] - self.firstNeedle[2] < 0:
            text = f'Withdraw first Needle with {abs(BestAIPoint[2] - self.firstNeedle[2])} mm'
            self.showPopupMessage2(text)
            print(text)
        if BestAIPoint[2] - self.firstNeedle[2] > 0:
            text = f'Push first Needle forward with {abs(BestAIPoint[2] - self.firstNeedle[2])} mm'
            self.showPopupMessage2(text)  
            print(text)
            
        if self.placement == 'Third placement':     
            BestAIPoint = self.top5_points[BestIndex][2]
    
            if BestAIPoint[2] - self.SecondNeedle[2] == 0:
                text = 'second needle is at the correct depth'
                self.showPopupMessage2(text)
                print(text)
            if BestAIPoint[2] - self.SecondNeedle[2] < 0:
                text = f'Withdraw second Needle with {abs(BestAIPoint[2] - self.SecondNeedle[2])} mm'
                self.showPopupMessage2(text)
                print(text)
            if BestAIPoint[2] - self.SecondNeedle[2] > 0:
                text = f'Push second Needle forward with {abs(BestAIPoint[2] - self.SecondNeedle[2])} mm'
                self.showPopupMessage2(text)  
                print(text)
            
        for i in range(len(BestIndices)):
            print('Bestindex', BestIndices[i])
            print('Best point location 1', self.top5_points[BestIndices[i]][1])
            print('Best point location 2', self.top5_points[BestIndices[i]][2])
            print('Best point overlap percentage', overlap[BestIndices[i]])
            print('Best point ablation distance', ablation[BestIndices[i]])  
        self.Best.setStyleSheet("background-color: green; color: white;")
        
        if self.placement == 'Second placement':
            if BestIndex > 0:
                iceballNode = slicer.util.getNode('IceballPred_' +str(BestIndex))
            else:
                iceballNode = slicer.util.getNode('IceballPred')
            volumeNode = self.inputSelector.currentNode()
            
            slicer.util.setSliceViewerLayers(foreground= iceballNode, background=volumeNode, foregroundOpacity=0.5)
            slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
        
                    
                    
        if self.placement == 'Third placement':
            iceballNode = slicer.util.getNode('IceballPred_' +str(BestIndex+5))
            volumeNode = self.inputSelector.currentNode()
            
            slicer.util.setSliceViewerLayers(foreground= iceballNode, background=volumeNode, foregroundOpacity=0.5)
            slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
            
        markupsNode = slicer.util.getNode("Best AI generated Point")
        self.focusSlicesOnPoints(markupsNode)
        
        
        for i in range(17):
            # Construct the segmentation node name
            segmentationNodeName = "Segmentation_" + str(i)
        
            try:
                # Retrieve the segmentation node by its name
                segmentationNode = slicer.util.getNode(segmentationNodeName)
            except slicer.util.MRMLNodeNotFoundException:
            
                continue  # Skip to the next node if not found
        
            # Get the display node of the segmentation
            displayNode = segmentationNode.GetDisplayNode()
            if displayNode:
                # Set visibility to false to hide it in the 3D and 2D views
                displayNode.SetVisibility(False)
                displayNode.SetVisibility2D(False)  # Optional: hide in slice views as well

                  
    def resample_volume_to_align(self, reference_volume, target_volume):
        """resamples target_volume to gain the same size as the reference_volume"""
        reference_volume = sitkUtils.PullVolumeFromSlicer(reference_volume)
        target_volume = sitkUtils.PullVolumeFromSlicer(target_volume)
        # Resample `target_volume` to align with `reference_volume` using nearest neighbor interpolation
        resampler = sitk.ResampleImageFilter()
        
        # Set reference image properties
        resampler.SetReferenceImage(reference_volume)
        resampler.SetOutputSpacing(reference_volume.GetSpacing())
        resampler.SetOutputOrigin(reference_volume.GetOrigin())
        resampler.SetOutputDirection(reference_volume.GetDirection())
        resampler.SetSize(reference_volume.GetSize())
        
        # Use nearest neighbor interpolation for binary data
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        
        # Perform resampling
        resampled_image = resampler.Execute(target_volume)
        resampled_node = sitkUtils.PushVolumeToSlicer(resampled_image, name="ResampledTarget")
        
        return resampled_node

        
    def calculate_ablation_margin(self, volume1, volume2):
        """Calculated ablation margin, defined as the smalles distance between the two volumes"""
        volume2 = self.resample_volume_to_align(volume1, volume2)
        # Extract boundary points of the binary volume
        def get_boundary_points(volumeNode):
            imageData = volumeNode.GetImageData()
            
            # Use Marching Cubes to extract the surface mesh
            surfaceFilter = vtk.vtkMarchingCubes()
            surfaceFilter.SetInputData(imageData)
            surfaceFilter.ComputeNormalsOff()
            surfaceFilter.SetValue(0, 0.5)  # 0.5 threshold for binary volumes
            surfaceFilter.Update()
            
            # Extract points from the surface
            polydata = surfaceFilter.GetOutput()
            points = vtk.vtkPoints()
            points.DeepCopy(polydata.GetPoints())
            
            # Convert to numpy array
            return np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        
        # Get boundary points for both volumes
        boundary_points1 = get_boundary_points(volume1)
        boundary_points2 = get_boundary_points(volume2)
    
        # Ensure there are points to compare
        if boundary_points1.size == 0 or boundary_points2.size == 0:
            raise ValueError("One of the volumes has no boundary points!")
    
        # Use KDTree for fast distance computation
        tree1 = KDTree(boundary_points1)
        distances, _ = tree1.query(boundary_points2)
    
        # Return the minimum distance
        return np.min(distances)
     
    def calculate_AI_Iceball_Tumor_overlap(self, segmentationNode1, segmentationNode2):
        """Calculates how much of the tumor is covered by the iceball"""
        allignedV2 = self.resample_volume_to_align(segmentationNode1, segmentationNode2) 
       #  Convert segmentation volumes to numpy arrays
        array1 = slicer.util.arrayFromVolume(segmentationNode1)
        array2 = slicer.util.arrayFromVolume(allignedV2)

        # Ensure both volumes have the same dimensions
        if array1.shape != array2.shape:
            raise ValueError("The segmentation volumes must have the same dimensions.")
        
        # Convert numpy arrays to PyTorch tensors and move to GPU
        tensor1 = torch.tensor(array1, dtype=torch.bool, device='cuda')
        tensor2 = torch.tensor(array2, dtype=torch.bool, device='cuda')
        
        # Calculate overlap (logical AND)
        overlap = tensor1 & tensor2  # Logical AND operation in PyTorch
        
        # Count the number of overlapping voxels
        overlap_voxels = overlap.sum().item()

        
        # Calculate overlap percentage (relative to either volume)
        volume1_voxels = tensor1.sum().item()
        overlap_percentage1 = (overlap_voxels / volume1_voxels) * 100 if volume1_voxels > 0 else 0

        return(overlap_percentage1)
        

# PredictionLogic
#

class PredictionLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")
      
  def createNeedleTubes(self,pos1):
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(2)
    pts.SetPoint(0, pos1[0], pos1[1], pos1[2])
    pts.SetPoint(1, pos1[0], pos1[1], pos1[2]-18)
    
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)


    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)

    tubes = vtk.vtkTubeFilter()
    tubes.SetInputData(poly)
    tubes.CappingOn()
    tubes.SidesShareVerticesOff()
    tubes.SetNumberOfSides(16)
    tubes.SetRadius(1.0)
    tubes.Update()
    
    return tubes
 
 
  def createProbeMaskList(self,inputVolume,image,center,nOfProbe):
  
    inputVolume = slicer.util.loadVolume('/home/snr/Documents/resampledNode.nrrd')
    RASToIJKMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetRASToIJKMatrix(RASToIJKMatrix)
    IjkToRasMatrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(IjkToRasMatrix)

    modelsLogic = slicer.modules.models.logic()
    segmentationNode = slicer.vtkMRMLSegmentationNode()
    slicer.mrmlScene.AddNode(segmentationNode)
    segmentationNode.CreateDefaultDisplayNodes() 
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
    for i in range(0,nOfProbe):
      position = [center[i][0], center[i][1], center[i][2], 1]
      #center_probe = RASToIJKMatrix.MultiplyPoint(position)
      pos1 = [position[0], position[1], position[2]]
      tubes = self.createNeedleTubes(pos1)  
      tubes.Update()
      model = modelsLogic.AddModel(tubes.GetOutput())
      nameNeedle = "needle-"+str(i)
      model.SetName(nameNeedle)
      model.SetDisplayVisibility(1)
      Needle1SegmentId = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(model.GetPolyData(), nameNeedle,
                                                                                     [0.0, 1.0, 0.0])                                                                           
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapVolumeNode.SetName("mask-needle")
    ids = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(ids)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, ids,labelmapVolumeNode, inputVolume)
    
    return sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(labelmapVolumeNode.GetName()),sitk.sitkFloat32) 
      
  def createProbeMask(self,inputVolume,image,probeLocation):
    inputVolume = slicer.util.loadVolume('/home/snr/Documents/resampledNode.nrrd')

    
    nOfPoint = probeLocation.GetNumberOfControlPoints()
    pos1 = [0, 0, 0]
    probeLocation.GetNthControlPointPosition(0, pos1)
    
    modelsLogic = slicer.modules.models.logic()
    segmentationNode = slicer.vtkMRMLSegmentationNode()
    slicer.mrmlScene.AddNode(segmentationNode)
    segmentationNode.CreateDefaultDisplayNodes() 
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
   
    for i in range(0,nOfPoint):
      probeLocation.GetNthControlPointPosition(i, pos1)
      tubes = self.createNeedleTubes(pos1)  
      tubes.Update()
      model = modelsLogic.AddModel(tubes.GetOutput())
      nameNeedle = "needle-"+str(i)
      model.SetName(nameNeedle)
      model.SetDisplayVisibility(0)
      Needle1SegmentId = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(model.GetPolyData(), nameNeedle,
                                                                                     [0.0, 1.0, 0.0])                                                                      
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapVolumeNode.SetName("mask-needle")
    ids = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(ids)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, ids,labelmapVolumeNode, inputVolume)
    
    return sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(labelmapVolumeNode.GetName()),sitk.sitkFloat32)
    
    
  def createInputImage(self,placementImage,labelmapNeedle,urethra):
    outputImageDir = '/home/snr/Documents/input/PC000-Input.nii.gz' 
    finalOrigin = placementImage.GetOrigin()
    finalSpacing = placementImage.GetSpacing()
    finalDirection = placementImage.GetDirection()

                   
    placementImage.SetOrigin([0] * 3)
    placementImage.SetSpacing([1] * 3)
    placementImage.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
    #needles
    labelmapNeedle.SetOrigin([0] * 3)
    labelmapNeedle.SetSpacing([1] * 3)
    labelmapNeedle.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
    #urethra
    urethra.SetOrigin([0] * 3)
    urethra.SetSpacing([1] * 3)
    urethra.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
    
    
    labelmapNeedle = sitk.Cast(labelmapNeedle, sitk.sitkFloat32)
    urethra = sitk.Cast(urethra, sitk.sitkFloat32)

    newImage = placementImage+labelmapNeedle*2000.0+urethra*-1100.0

    newImage.SetOrigin(finalOrigin)
    newImage.SetSpacing(finalSpacing)
    newImage.SetDirection(finalDirection)

    
    sitk.WriteImage(newImage, outputImageDir)
    
  def resampleImage(self,srcImage, refImage, transform, interp='linear'):
    
    dimension = srcImage.GetDimension()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(refImage)
    if interp == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(refImage.GetSpacing())
    resampler.SetSize(refImage.GetSize())
    resampler.SetOutputDirection(srcImage.GetDirection())
    resampler.SetOutputOrigin(srcImage.GetOrigin())
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(0)
    
    dstImage = resampler.Execute(srcImage)
    
    return dstImage

  def outsideProcess(self, inputVolume, outputVolume, probeLocation,urethraLabel,nOfProbes):
  
    base = '/home/snr/Downloads/ToEva/IceballPrediction/Resources/PC001-Input.nii.gz'
    
    base = '/home/snr/Downloads/ToEva/IceballPredicnput.nii.gz'
    baseImage = sitk.ReadImage(base, sitk.sitkFloat32)

    placementImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(inputVolume.GetName()),sitk.sitkFloat32)

    transform = sitk.Transform()
    ablationImageResampled = self.resampleImage(placementImage, baseImage, transform, interp='linear')
    
    resampledImagePath = '/home/snr/Documents/resampledNode.nrrd'
    sitk.WriteImage(ablationImageResampled, resampledImagePath)
   
    labelmapNeedle = self.createProbeMaskList(inputVolume,ablationImageResampled,probeLocation,nOfProbes)
   
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapVolumeNode.SetName("mask-urethra")
    ids = vtk.vtkStringArray()
    urethraLabel.GetDisplayNode().GetVisibleSegmentIDs(ids)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(urethraLabel, ids,labelmapVolumeNode, inputVolume)
    
    labelmapUrethra = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(labelmapVolumeNode.GetName()),sitk.sitkFloat32)  
    maskedImage = self.createInputImage(ablationImageResampled,labelmapNeedle,labelmapUrethra)
  
    
   
    home_dir = os.system("cd ~")
    re = os.system("PythonSlicer /home/snr/Downloads/ToEva/monitorfolder.py")
    
    
    outname = '/home/snr/Documents/output/PC000-Input_seg.nii.gz'
    outImage = sitk.ReadImage(outname)

    filter = sitk.BinaryErodeImageFilter()
    filter.SetKernelRadius(1)
    filter.SetForegroundValue(1)
    eroded = filter.Execute(outImage)

    filter2 = sitk.BinaryDilateImageFilter()
    filter2.SetKernelRadius(2)
    filter2.SetForegroundValue(1)
    eroded = filter2.Execute(eroded)
    
    sitkUtils.PushVolumeToSlicer(eroded, targetNode=None, name="pred_iceball", className='vtkMRMLScalarVolumeNode')
    return eroded
   
  def process(self, inputVolume, outputVolume, probeLocation,urethraLabel):
  
    base = '/home/snr/Downloads/ToEva/IceballPrediction/Resources/PC001-Input.nii.gz'
    baseImage = sitk.ReadImage(base, sitk.sitkFloat32)

    placementImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(inputVolume.GetName()),sitk.sitkFloat32)

    transform = sitk.Transform()
    ablationImageResampled = self.resampleImage(placementImage, baseImage, transform, interp='linear')
    
    resampledImagePath = '/home/snr/Documents/resampledNode.nrrd'
    sitk.WriteImage(ablationImageResampled, resampledImagePath)
   
    labelmapNeedle = self.createProbeMask(inputVolume,ablationImageResampled,probeLocation)
   
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmapVolumeNode.SetName("mask-urethra")
    ids = vtk.vtkStringArray()
    urethraLabel.GetDisplayNode().GetVisibleSegmentIDs(ids)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(urethraLabel, ids,labelmapVolumeNode, inputVolume)
    
    labelmapUrethra = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(labelmapVolumeNode.GetName()),sitk.sitkFloat32)
    
    
    labelmapUrethra = self.resampleImage(labelmapUrethra, baseImage, transform, interp='linear')
    

    maskedImage = self.createInputImage(ablationImageResampled,labelmapNeedle,labelmapUrethra)
    
    
    start = time.time()


    home_dir = os.system("cd ~")

    re = os.system("PythonSlicer /home/snr/Downloads/ToEva/monitorfolder.py")
    
    outname = '/home/snr/Documents/output/PC000-Input_seg.nii.gz'
    outImage = sitk.ReadImage(outname)

    filter = sitk.BinaryErodeImageFilter()
    filter.SetKernelRadius(1)
    filter.SetForegroundValue(1)
    eroded = filter.Execute(outImage)

    filter2 = sitk.BinaryDilateImageFilter()
    filter2.SetKernelRadius(2)
    filter2.SetForegroundValue(1)
    eroded = filter2.Execute(eroded)
    
    sitkUtils.PushVolumeToSlicer(eroded, targetNode=None, name="IceballPred", className='vtkMRMLScalarVolumeNode')
    end = time.time()


