import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
# This is a python fork of "Fast Bounded Power Diagram", which is developed originaly in matlab
# (by Muhammad Kasim, muhammad.kasim@wolfson.ox.ac.uk).
# the original matlab version is available on:
# https://www.mathworks.com/matlabcentral/fileexchange/56633-fast-bounded-power-diagram
#
class LaguerreDiagram(object): # power diagram also called laguerre diagram (LD)
    def __init__(self,S,wts,bound_vertexes=None,bounded=True):
        '''
        This function computes the power cells using the points in S
        the bounding box (must be a rectangle) must be used by defining its vertexes.
        If bound_vertexes is not supplied, the axis-aligned box of S is used.
        Input:
        * S: coordinate of the Voronoi point (numPoints x 1)
        * wts: weights of each point (numPoints x 1)
        * bound_vertexes: vortices of the bounding box in cw order (numVert x 2)
        attributes:
        * all above mentioned inputs
        * V: x,y-coordinate of vertices of the power cells
        * CE: indices of the Voronoi cells from V
        (-1 if the corresponding voronoi point are enclosed by other points)
        * Sused: coordinate of the Voronoi point used in cell generation
        * wtsused: weights used in cell generation
        ########################################################
        Note that the core of this fucintion is originally developed in matlab by
        Aaron Becker (atbecker@uh.edu)
        and Muhammad (Kasim muhammad.kasim@wolfson.ox.ac.uk)
        The cell cropping process, however, is discarded,
        and the reflection method is used to solve the problem of having cells exceed boundaries.
        This little modification is done by LIANG Zhengyu (liangzhengyu@csu.edu.cn)
        ''' 
        S = np.asarray(S)
        wts = np.asarray(wts)
        # assure that S and wts have the right shapes
        if (len(S.shape) != 2):
            raise ValueError('S must be a 2D matrix')
        elif (S.shape[1] != 2):
            raise ValueError('S must be a N*2 matrix')
            
        if (len(wts.shape) !=2):
            raise ValueError('wts must be a 2D matrix')
        elif (wts.shape[1] != 1) or (wts.shape[0] != S.shape[0]):
            raise ValueError('wts must be a N*1 vector that has the same row number as S')
        if bound_vertexes is None:
            # using the bounding box as the boundary (a smooth value is added to assure that every 
            # point has its own symmetric point, with respect to the boundary)
            bnd=[S[:,0].min()-1e-3,S[:,0].max()+1e-3,S[:,1].min()-1e-3,S[:,1].max()+1e-3]
            bound_vertexes = np.asarray([[bnd[0],bnd[3]],[bnd[1],bnd[3]],[bnd[1],bnd[2]],[bnd[0],bnd[2]]])
        elif (bound_vertexes.shape[1] != 2) or (len(bound_vertexes.shape) != 2) or (bound_vertexes.shape[0] !=4):
            raise ValueError('if supplied, bound_vertexes must be a 4*2 vector')
        # initiating attributes
        self.bounded = bounded # see if bounded mode is used
        self.V = None
        self.CE = None
        self.S = S
        self.wts = wts
        self.bound_vertexes = bound_vertexes
        del S,wts,bounded,bound_vertexes
        # exclude points outside the boundary
        indomain = (self.S[:,0] >= self.bound_vertexes[:,0].min()) \
         & (self.S[:,0] <= self.bound_vertexes[:,0].max()) \
         & (self.S[:,1] >= self.bound_vertexes[:,1].min()) \
         & (self.S[:,1] <= self.bound_vertexes[:,1].max())
        if ~indomain.all():
            self.Sused = self.S[indomain,:]
            self.wtsused = self.wts[indomain,:]
            exculnum = self.S.shape[0]- self.Sused.shape[0]
            print('--------------------------------------------------------------')
            print('%d points have been excluded, as they are outside the boundary' %(exculnum))
        else:
            self.Sused = self.S
            self.wtsused = self.wts
        # compute initial LD
        tempVi,tempCe = self._powerDiagram2(self.Sused,self.wtsused,self.bound_vertexes)
        
        if self.bounded:
            V,CE = self._boundedLD(tempVi,tempCe,self.Sused,self.wtsused)
        else:
            V = tempVi
            CE =tempCe
        V,CE = self._sortcells(V,CE)
        self.V = V
        self.CE = CE


    def _powerDiagram2(self,S,wts,bound_vertexes):
        '''
        Input:
        * S: a matrix that specifies the sites coordinates (Npts x 2)
        * wts: a column vector that specifies the sites' weights (Npts x 1)
        Output:
        * V: list of points' coordinates that makes the power diagram vertices.
        * CE: cells that contains index of coordinate in V that makes the power diagram of a specified site.
        ########################################################
        For algorithms, read:
        [1] Aurenhammer, & F. (1987). Power diagrams: properties, algorithms and applications.
        SIAM Journal on Computing, 16(1), 78-96.
        [2]  Nocaj, Arlind , and U. Brandes . "Computing Voronoi Treemaps: Faster, Simpler,
        and Resolution-independent." Computer Graphics Forum 31.3pt1(2012):855-864.
        '''
        ### prepare points for LD generation ###
        # the length and the width of provided boundary            
        rgx = bound_vertexes[:,0].max() - bound_vertexes[:,0].min()
        rgy = bound_vertexes[:,1].max() - bound_vertexes[:,1].min()
        rg = np.max([rgx,rgy])
        midx = (bound_vertexes[:,0].max() - bound_vertexes[:,0].min())/2
        midy = (bound_vertexes[:,1].max() - bound_vertexes[:,1].min())/2
        # add 4 additional edges to be the upper hull in powerDiagram2        
        xA = np.vstack((S[:,0,np.newaxis], midx + np.asarray([[0],[0],[-5*rg],[5*rg]])))
        yA = np.vstack((S[:,1,np.newaxis], midy + np.asarray([[-5*rg],[+5*rg],[0],[0]])))
        wtsA = np.vstack((wts,np.zeros((4,1))))
        S = np.hstack((xA,yA))
        wts = wtsA
        ### LD generation ####
        # calculate the poles of sites' corresponding planes
        S_3 = np.sum(S**2,axis=1,keepdims=True)-wts
        S = np.hstack((S,S_3))     # this operation maps S to the dual space
        # get the 3D convhull
        hull = ConvexHull(S)
        C = np.asarray(hull.simplices)
        norms = - hull.equations[:,0:-1]
        lowerIdx = norms[:,-1] > 0
        upperIdx = norms[:,-1] <= 0
        # label the lower/upper hull
        CUp = C[upperIdx,:]
        C = C[lowerIdx,:]
        norms = norms[lowerIdx,:]
        # midPoints = midPoints[lowerIdx,:]
        Zs = -norms[:,-1,np.newaxis]
        normsZ = norms/Zs
        # use polarity function to map the facets to real space #
        V = np.hstack((normsZ[:,0,np.newaxis]/2, normsZ[:,1,np.newaxis]/2)) # this is excatly the vertices for LD
        #V = np.vstack((np.asarray([float("inf"),float("inf")]),V)) # add infinity elements to store upper hull's vertices.
        CE = list(-np.ones((S.shape[0],1))) # initialize a list to store LD cells
        for col in range(0,C.shape[1]):
            for row in range(0,C.shape[0]):
                i = C[row,col]
                if CE[i][0] == -1:
                    CE[i] = np.ones((1),dtype=np.int32)*(row)
                else:
                    CE[i] = np.hstack((CE[i], row))
        
        for col in range(0,CUp.shape[1]):
            for row in range(0,CUp.shape[0]):
                i = CUp[row,col]
                if CE[i][0] == -1:
                    CE[i] = np.zeros((1),dtype=np.int32) # assign inf to upper hull's vertices.
                elif CE[i].min() > 0:
                    CE[i] = np.hstack((CE[i], 0))
        # remove the last 4 cells to avoid inf values
        CE = CE[0:-4]
        return V, CE

    # utilities
    def showLD(self):
        # draw cells
        for singleCE in self.CE:
            if (singleCE == -1).any():
                continue
            plt.plot(np.hstack((self.V[singleCE, 0],self.V[singleCE[0], 0 ,np.newaxis])),\
                np.hstack((self.V[singleCE, 1],self.V[singleCE[0], 1 ,np.newaxis])), 'k-')
            # draw vertices
            plt.plot(self.V[singleCE,0], self.V[singleCE,1], 'r.')
        # draw boundary
        plt.plot(np.hstack((self.bound_vertexes[:,0],self.bound_vertexes[0,0,np.newaxis])) \
                ,np.hstack((self.bound_vertexes[:,1],self.bound_vertexes[0,1,np.newaxis])), \
                '--b')
        # draw sites
        plt.plot(self.S[:,0],self.S[:,1],'k.')
        plt.axis('equal')
        plt.show()
    def countcells(self):
        '''
        Output effective cell num
        '''
        invalidnum = 0
        for CEi in self.CE:
            if (CEi == -1).any():
                invalidnum += 1
        return len(self.CE)-invalidnum
    def _cart2pol(self,xy):
        """
        mimic the matlab cart2pol function
        """
        r = np.sqrt(xy[:,0]**2 + xy[:,1]**2)
        t = np.arctan2(xy[:,1],xy[:,0])
        t[t<0] += np.pi*2
        return t,r

    def _sortcells(self,V,CE):
        """
        sort vertexes to make them be arranged counter clockwise
        """
        for i in range(0,len(CE)):
            CEi = CE[i]
            if (CEi == -1).any():
                continue
            pts = V[CEi,:]
            t,_ = self._cart2pol(pts-np.mean(pts,axis=0,keepdims=True))
            order = np.argsort(t)
            CE[i] = CEi[order]
        return V,CE
    def _boundedLD(self,tempVi,tempCe,S,wt):
        '''
        apply mollon's algorithm to generate bounded laguerre diagram
        see:
        [1]	MOLLON G, ZHAO J. Fourier–Voronoi-based generation of realistic 
        samples for discrete modelling of granular materials [J]. Granular Matter, 2012, 14(5): 621-38.
        '''
        minx = self.bound_vertexes[:,0].min()
        maxx = self.bound_vertexes[:,0].max()
        miny = self.bound_vertexes[:,1].min()
        maxy = self.bound_vertexes[:,1].max()
        symmetric_sites = np.asarray([[float("inf"),float("inf")]])
        symmetric_weights = np.asarray([[float("inf")]])
        for i in range(0,len(tempCe)):

            cell = tempCe[i]
            if not (cell == -1).any():# if it is removed cells
                vetx = tempVi[cell,:]
                Site = S[i,:]
                weight = wt[i,:]
                # if it is problematic cell
                if (vetx[:,0] < minx).any():
                    tsym_s = 2*np.asarray([minx,Site[1]])-Site
                    symmetric_sites = np.vstack((symmetric_sites,tsym_s))
                    symmetric_weights =  np.vstack((symmetric_weights,weight))
                if (vetx[:,0] > maxx).any():
                    tsym_s = 2*np.asarray([maxx,Site[1]])-Site
                    symmetric_sites = np.vstack((symmetric_sites,tsym_s))
                    symmetric_weights =  np.vstack((symmetric_weights,weight))
                if (vetx[:,1] < miny).any():
                    tsym_s = 2*np.asarray([Site[0],miny])-Site
                    symmetric_sites = np.vstack((symmetric_sites,tsym_s))
                    symmetric_weights =  np.vstack((symmetric_weights,weight))
                if (vetx[:,1] > maxy).any():
                    tsym_s = 2*np.asarray([Site[0],maxy])-Site
                    symmetric_sites = np.vstack((symmetric_sites,tsym_s))
                    symmetric_weights =  np.vstack((symmetric_weights,weight))
        # merge symmetric sites with original sites
        newS = np.vstack((S,symmetric_sites[1:,:]))
        newWt = np.vstack((wt,symmetric_weights[1:,:]))
        symmetric_num = symmetric_sites.shape[0]-1
        newV,newCE = self._powerDiagram2(newS,newWt,self.bound_vertexes)
        newCE = newCE[0:-symmetric_num]
        return newV, newCE

    ## shape evaluation
    def cell_aixs(self,downsampling = False,samplenum = 100):
        '''
        computing all cells' axis
        '''
        if not self.bounded:
            minx = self.bound_vertexes[:,0].min()
            maxx = self.bound_vertexes[:,0].max()
            miny = self.bound_vertexes[:,1].min()
            maxy = self.bound_vertexes[:,1].max()
        if downsampling:
            indexes = np.random.choice (len(self.CE),samplenum)
        else:
            indexes = range(0,len(self.CE))
        Ls = -np.ones(len(indexes))
        Ss = -np.ones(len(indexes))
        LAs = -np.ones(len(indexes))
        count = 0
        for i in indexes:
            cell = self.CE[i]
            if (cell == -1).any():
                count += 1
                continue
            poly = self.V[cell]
            if not self.bounded:
                if poly[:,0].max()>maxx or poly[:,0].min()<minx \
                or poly[:,1].max()>maxy or poly[:,1].min()<miny:
                    count += 1
                    continue
            L,S,LA = self._polyFeret_2D(poly)
            Ls[count] = L
            Ss[count] = S
            LAs[count] = LA
            count += 1
        return Ls,Ss,LAs
    
    def cell_areas(self,downsampling = False,samplenum = 100):
        '''
        computing all cells' area
        '''
        if not self.bounded:
            minx = self.bound_vertexes[:,0].min()
            maxx = self.bound_vertexes[:,0].max()
            miny = self.bound_vertexes[:,1].min()
            maxy = self.bound_vertexes[:,1].max()
        if downsampling:
            indexes = np.random.choice (len(self.CE),samplenum)
        else:
            indexes = range(0,len(self.CE))
        areas = -np.ones(len(indexes))
        count = 0
        for i in indexes:
            cell = self.CE[i]
            if (cell == -1).any():
                count += 1
                continue
            poly = self.V[cell]
            if not self.bounded:
                if poly[:,0].max()>maxx or poly[:,0].min()<minx \
                or poly[:,1].max()>maxy or poly[:,1].min()<miny:
                    count += 1
                    continue
            area = self._polyarea(poly)
            areas[count] = area
            count += 1
        return areas

    def cell_perimeters(self,downsampling = False,samplenum = 100):
        pass
    
    ## shape functions
    def _polyarea(self,xy):
        '''
        computing the area of a polygon, which is supplied
        '''
        xya = xy[0:,:]
        xyb = np.vstack((xy[1:,:],xy[:1,:]))
        area = np.abs(np.sum(np.cross(xya,xyb))/2)
        return area
    
    def _polyFeret_2D(self,xy):
        # normalization
        xy = xy-np.mean(xy,axis=0)
        # rotation angles
        angles = np.arange(0,np.pi,np.pi/180)
        lengthx = np.zeros((angles.shape[0]))
        lengthy = np.zeros((angles.shape[0]))
        # assemble rotation matrix
        angles3 = np.reshape(angles,(-1,1,1))
        Rm1 = np.cos(angles3)
        Rm2 = np.sin(angles3)
        Rmrow1 = np.concatenate((Rm1,Rm2),axis=2)
        Rmrow2 = np.concatenate((-Rm2,Rm1),axis=2)
        Rm = np.concatenate((Rmrow1,Rmrow2),axis=1)
        # rotation
        xy3 = np.repeat(xy[np.newaxis,:,:],angles.shape[0],axis=0)
        xy3r = np.matmul(xy3, Rm)
        # find shortest axis
        lengthx = np.amax(xy3r[:,:,0],axis = -1) - np.amin(xy3r[:,:,0],axis = -1)
        lengthy = np.amax(xy3r[:,:,1],axis = -1) - np.amin(xy3r[:,:,1],axis = -1)        
        S = lengthx.min()
        minI = np.argmin(lengthx)
        # find long axis and orientation accordingly
        L = lengthy[minI]
        LA = np.pi/2 - angles[minI]
        if LA<0:
            LA += np.pi
        return L,S,LA

    def _polyperimeter(self,xy):
        pass
